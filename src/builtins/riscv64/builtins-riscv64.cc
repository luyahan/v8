// Copyright 2012 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#if V8_TARGET_ARCH_RISCV64

#include "src/api/api-arguments.h"
#include "src/codegen/code-factory.h"
#include "src/debug/debug.h"
#include "src/deoptimizer/deoptimizer.h"
#include "src/execution/frame-constants.h"
#include "src/execution/frames.h"
#include "src/logging/counters.h"
// For interpreter_entry_return_pc_offset. TODO(jkummerow): Drop.
#include "src/codegen/macro-assembler-inl.h"
#include "src/codegen/register-configuration.h"
#include "src/codegen/riscv64/constants-riscv64.h"
#include "src/heap/heap-inl.h"
#include "src/objects/cell.h"
#include "src/objects/foreign.h"
#include "src/objects/heap-number.h"
#include "src/objects/js-generator.h"
#include "src/objects/objects-inl.h"
#include "src/objects/smi.h"
#include "src/runtime/runtime.h"
#include "src/wasm/wasm-linkage.h"
#include "src/wasm/wasm-objects.h"

namespace v8 {
namespace internal {

#define __ ACCESS_MASM(masm)

void Builtins::Generate_Adaptor(MacroAssembler* masm, Address address) {
  __ RecordComment(
      "[  li(kJavaScriptCallExtraArg1Register, "
      "ExternalReference::Create(address));");
  __ li(kJavaScriptCallExtraArg1Register, ExternalReference::Create(address));
  __ RecordComment("]");
  __ Jump(BUILTIN_CODE(masm->isolate(), AdaptorWithBuiltinExitFrame),
          RelocInfo::CODE_TARGET);
}

static void GenerateTailCallToReturnedCode(MacroAssembler* masm,
                                           Runtime::FunctionId function_id) {
  // ----------- S t a t e -------------
  //  -- a0 : actual argument count
  //  -- a1 : target function (preserved for callee)
  //  -- a3 : new target (preserved for callee)
  // -----------------------------------
  {
    FrameScope scope(masm, StackFrame::INTERNAL);
    // Push a copy of the target function, the new target and the actual
    // argument count.
    // Push function as parameter to the runtime call.
    __ RecordComment("[  SmiTag(kJavaScriptCallArgCountRegister);");
    __ SmiTag(kJavaScriptCallArgCountRegister);
    __ RecordComment("]");
    __ Push(kJavaScriptCallTargetRegister, kJavaScriptCallNewTargetRegister,
            kJavaScriptCallArgCountRegister, kJavaScriptCallTargetRegister);

    __ RecordComment("[  CallRuntime(function_id, 1);");
    __ CallRuntime(function_id, 1);
    __ RecordComment("]");
    // Use the return value before restoring a0
    __ RecordComment(
        "[  Add64(a2, a0, Operand(Code::kHeaderSize - kHeapObjectTag));");
    __ Add64(a2, a0, Operand(Code::kHeaderSize - kHeapObjectTag));
    __ RecordComment("]");
    // Restore target function, new target and actual argument count.
    __ Pop(kJavaScriptCallTargetRegister, kJavaScriptCallNewTargetRegister,
           kJavaScriptCallArgCountRegister);
    __ RecordComment("[  SmiUntag(kJavaScriptCallArgCountRegister);");
    __ SmiUntag(kJavaScriptCallArgCountRegister);
    __ RecordComment("]");
  }

  static_assert(kJavaScriptCallCodeStartRegister == a2, "ABI mismatch");
  __ RecordComment("[  Jump(a2);");
  __ Jump(a2);
  __ RecordComment("]");
}

namespace {

enum StackLimitKind { kInterruptStackLimit, kRealStackLimit };

void LoadStackLimit(MacroAssembler* masm, Register destination,
                    StackLimitKind kind) {
  DCHECK(masm->root_array_available());
  Isolate* isolate = masm->isolate();
  ExternalReference limit =
      kind == StackLimitKind::kRealStackLimit
          ? ExternalReference::address_of_real_jslimit(isolate)
          : ExternalReference::address_of_jslimit(isolate);
  DCHECK(TurboAssembler::IsAddressableThroughRootRegister(isolate, limit));

  intptr_t offset =
      TurboAssembler::RootRegisterOffsetForExternalReference(isolate, limit);
  CHECK(is_int32(offset));
  __ RecordComment(
      "[  Ld(destination, MemOperand(kRootRegister, "
      "static_cast<int32_t>(offset)));");
  __ Ld(destination, MemOperand(kRootRegister, static_cast<int32_t>(offset)));
  __ RecordComment("]");
}

void Generate_JSBuiltinsConstructStubHelper(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0     : number of arguments
  //  -- a1     : constructor function
  //  -- a3     : new target
  //  -- cp     : context
  //  -- ra     : return address
  //  -- sp[...]: constructor arguments
  // -----------------------------------

  // Enter a construct frame.
  {
    FrameScope scope(masm, StackFrame::CONSTRUCT);

    // Preserve the incoming parameters on the stack.
    __ RecordComment("[  SmiTag(a0);");
    __ SmiTag(a0);
    __ RecordComment("]");
    __ RecordComment("[  Push(cp, a0);");
    __ Push(cp, a0);
    __ RecordComment("]");
    __ RecordComment("[  SmiUntag(a0);");
    __ SmiUntag(a0);
    __ RecordComment("]");

    // Set up pointer to last argument (skip receiver).
    __ Add64(
        t2, fp,
        Operand(StandardFrameConstants::kCallerSPOffset + kSystemPointerSize));
    // Copy arguments and receiver to the expression stack.
    __ RecordComment("[  PushArray(t2, a0, t3, t0);");
    __ PushArray(t2, a0, t3, t0);
    __ RecordComment("]");
    // The receiver for the builtin/api call.
    __ RecordComment("[  PushRoot(RootIndex::kTheHoleValue);");
    __ PushRoot(RootIndex::kTheHoleValue);
    __ RecordComment("]");

    // Call the function.
    // a0: number of arguments (untagged)
    // a1: constructor function
    // a3: new target
    __ RecordComment(
        "[  InvokeFunctionWithNewTarget(a1, a3, a0, CALL_FUNCTION);");
    __ InvokeFunctionWithNewTarget(a1, a3, a0, CALL_FUNCTION);
    __ RecordComment("]");

    // Restore context from the frame.
    __ RecordComment(
        "[  Ld(cp, MemOperand(fp, ConstructFrameConstants::kContextOffset));");
    __ Ld(cp, MemOperand(fp, ConstructFrameConstants::kContextOffset));
    __ RecordComment("]");
    // Restore smi-tagged arguments count from the frame.
    __ RecordComment(
        "[  Ld(t3, MemOperand(fp, ConstructFrameConstants::kLengthOffset));");
    __ Ld(t3, MemOperand(fp, ConstructFrameConstants::kLengthOffset));
    __ RecordComment("]");
    // Leave construct frame.
  }

  // Remove caller arguments from the stack and return.
  __ RecordComment("[  SmiScale(t3, t3, kSystemPointerSizeLog2);");
  __ SmiScale(t3, t3, kSystemPointerSizeLog2);
  __ RecordComment("]");
  __ RecordComment("[  Add64(sp, sp, t3);");
  __ Add64(sp, sp, t3);
  __ RecordComment("]");
  __ RecordComment("[  Add64(sp, sp, kSystemPointerSize);");
  __ Add64(sp, sp, kSystemPointerSize);
  __ RecordComment("]");
  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");
}

static void Generate_StackOverflowCheck(MacroAssembler* masm, Register num_args,
                                        Register scratch1, Register scratch2,
                                        Label* stack_overflow) {
  // Check the stack for overflow. We are not trying to catch
  // interruptions (e.g. debug break and preemption) here, so the "real stack
  // limit" is checked.
  LoadStackLimit(masm, scratch1, StackLimitKind::kRealStackLimit);
  // Make scratch1 the space we have left. The stack might already be overflowed
  // here which will cause scratch1 to become negative.
  __ RecordComment("[  Sub64(scratch1, sp, scratch1);");
  __ Sub64(scratch1, sp, scratch1);
  __ RecordComment("]");
  // Check if the arguments will overflow the stack.
  __ RecordComment("[  Sll64(scratch2, num_args, kSystemPointerSizeLog2);");
  __ Sll64(scratch2, num_args, kSystemPointerSizeLog2);
  __ RecordComment("]");
  // Signed comparison.
  __ RecordComment(
      "[  Branch(stack_overflow, le, scratch1, Operand(scratch2));");
  __ Branch(stack_overflow, le, scratch1, Operand(scratch2));
  __ RecordComment("]");
}

}  // namespace

// The construct stub for ES5 constructor functions and ES6 class constructors.
void Builtins::Generate_JSConstructStubGeneric(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  --      a0: number of arguments (untagged)
  //  --      a1: constructor function
  //  --      a3: new target
  //  --      cp: context
  //  --      ra: return address
  //  -- sp[...]: constructor arguments
  // -----------------------------------

  // Enter a construct frame.
  {
    FrameScope scope(masm, StackFrame::CONSTRUCT);
    Label post_instantiation_deopt_entry, not_create_implicit_receiver;

    // Preserve the incoming parameters on the stack.
    __ RecordComment("[  SmiTag(a0);");
    __ SmiTag(a0);
    __ RecordComment("]");
    __ RecordComment("[  Push(cp, a0, a1);");
    __ Push(cp, a0, a1);
    __ RecordComment("]");
    __ RecordComment("[  PushRoot(RootIndex::kUndefinedValue);");
    __ PushRoot(RootIndex::kUndefinedValue);
    __ RecordComment("]");
    __ RecordComment("[  Push(a3);");
    __ Push(a3);
    __ RecordComment("]");

    // ----------- S t a t e -------------
    //  --        sp[0*kSystemPointerSize]: new target
    //  --        sp[1*kSystemPointerSize]: padding
    //  -- a1 and sp[2*kSystemPointerSize]: constructor function
    //  --        sp[3*kSystemPointerSize]: number of arguments (tagged)
    //  --        sp[4*kSystemPointerSize]: context
    // -----------------------------------

    __ RecordComment(
        "[  LoadTaggedPointerField(t2, FieldMemOperand(a1, "
        "JSFunction::kSharedFunctionInfoOffset));");
    __ LoadTaggedPointerField(
        t2, FieldMemOperand(a1, JSFunction::kSharedFunctionInfoOffset));
    __ RecordComment("]");
    __ RecordComment(
        "[  Lwu(t2, FieldMemOperand(t2, SharedFunctionInfo::kFlagsOffset));");
    __ Lwu(t2, FieldMemOperand(t2, SharedFunctionInfo::kFlagsOffset));
    __ RecordComment("]");
    __ RecordComment(
        "[  DecodeField<SharedFunctionInfo::FunctionKindBits>(t2);");
    __ DecodeField<SharedFunctionInfo::FunctionKindBits>(t2);
    __ RecordComment("]");
    __ JumpIfIsInRange(t2, kDefaultDerivedConstructor, kDerivedConstructor,
                       &not_create_implicit_receiver);

    // If not derived class constructor: Allocate the new receiver object.
    __ IncrementCounter(masm->isolate()->counters()->constructed_objects(), 1,
                        t2, t4);
    __ Call(BUILTIN_CODE(masm->isolate(), FastNewObject),
            RelocInfo::CODE_TARGET);
    __ RecordComment("[  Branch(&post_instantiation_deopt_entry);");
    __ Branch(&post_instantiation_deopt_entry);
    __ RecordComment("]");

    // Else: use TheHoleValue as receiver for constructor call
    __ RecordComment("[  bind(&not_create_implicit_receiver);");
    __ bind(&not_create_implicit_receiver);
    __ RecordComment("]");
    __ RecordComment("[  LoadRoot(a0, RootIndex::kTheHoleValue);");
    __ LoadRoot(a0, RootIndex::kTheHoleValue);
    __ RecordComment("]");

    // ----------- S t a t e -------------
    //  --                          a0: receiver
    //  -- Slot 4 / sp[0*kSystemPointerSize]: new target
    //  -- Slot 3 / sp[1*kSystemPointerSize]: padding
    //  -- Slot 2 / sp[2*kSystemPointerSize]: constructor function
    //  -- Slot 1 / sp[3*kSystemPointerSize]: number of arguments (tagged)
    //  -- Slot 0 / sp[4*kSystemPointerSize]: context
    // -----------------------------------
    // Deoptimizer enters here.
    masm->isolate()->heap()->SetConstructStubCreateDeoptPCOffset(
        masm->pc_offset());
    __ RecordComment("[  bind(&post_instantiation_deopt_entry);");
    __ bind(&post_instantiation_deopt_entry);
    __ RecordComment("]");

    // Restore new target.
    __ RecordComment("[  Pop(a3);");
    __ Pop(a3);
    __ RecordComment("]");

    // Push the allocated receiver to the stack.
    __ RecordComment("[  Push(a0);");
    __ Push(a0);
    __ RecordComment("]");

    // We need two copies because we may have to return the original one
    // and the calling conventions dictate that the called function pops the
    // receiver. The second copy is pushed after the arguments, we saved in a6
    // since a0 will store the return value of callRuntime.
    __ RecordComment("[  Move(a6, a0);");
    __ Move(a6, a0);
    __ RecordComment("]");

    // Set up pointer to last argument.
    __ Add64(
        t2, fp,
        Operand(StandardFrameConstants::kCallerSPOffset + kSystemPointerSize));

    // ----------- S t a t e -------------
    //  --                 r3: new target
    //  -- sp[0*kSystemPointerSize]: implicit receiver
    //  -- sp[1*kSystemPointerSize]: implicit receiver
    //  -- sp[2*kSystemPointerSize]: padding
    //  -- sp[3*kSystemPointerSize]: constructor function
    //  -- sp[4*kSystemPointerSize]: number of arguments (tagged)
    //  -- sp[5*kSystemPointerSize]: context
    // -----------------------------------

    // Restore constructor function and argument count.
    __ RecordComment(
        "[  Ld(a1, MemOperand(fp, "
        "ConstructFrameConstants::kConstructorOffset));");
    __ Ld(a1, MemOperand(fp, ConstructFrameConstants::kConstructorOffset));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(a0, MemOperand(fp, ConstructFrameConstants::kLengthOffset));");
    __ Ld(a0, MemOperand(fp, ConstructFrameConstants::kLengthOffset));
    __ RecordComment("]");
    __ RecordComment("[  SmiUntag(a0);");
    __ SmiUntag(a0);
    __ RecordComment("]");

    Label enough_stack_space, stack_overflow;
    Generate_StackOverflowCheck(masm, a0, t0, t1, &stack_overflow);
    __ RecordComment("[  Branch(&enough_stack_space);");
    __ Branch(&enough_stack_space);
    __ RecordComment("]");

    __ RecordComment("[  bind(&stack_overflow);");
    __ bind(&stack_overflow);
    __ RecordComment("]");
    // Restore the context from the frame.
    __ RecordComment(
        "[  Ld(cp, MemOperand(fp, ConstructFrameConstants::kContextOffset));");
    __ Ld(cp, MemOperand(fp, ConstructFrameConstants::kContextOffset));
    __ RecordComment("]");
    __ RecordComment("[  CallRuntime(Runtime::kThrowStackOverflow);");
    __ CallRuntime(Runtime::kThrowStackOverflow);
    __ RecordComment("]");
    // Unreachable code.
    __ RecordComment("[  break_(0xCC);");
    __ break_(0xCC);
    __ RecordComment("]");

    __ RecordComment("[  bind(&enough_stack_space);");
    __ bind(&enough_stack_space);
    __ RecordComment("]");

    // Copy arguments and receiver to the expression stack.
    __ RecordComment("[  PushArray(t2, a0, t0, t1);");
    __ PushArray(t2, a0, t0, t1);
    __ RecordComment("]");
    // We need two copies because we may have to return the original one
    // and the calling conventions dictate that the called function pops the
    // receiver. The second copy is pushed after the arguments,
    __ RecordComment("[  Push(a6);");
    __ Push(a6);
    __ RecordComment("]");

    // Call the function.
    __ RecordComment(
        "[  InvokeFunctionWithNewTarget(a1, a3, a0, CALL_FUNCTION);");
    __ InvokeFunctionWithNewTarget(a1, a3, a0, CALL_FUNCTION);
    __ RecordComment("]");

    // ----------- S t a t e -------------
    //  --                 a0: constructor result
    //  -- sp[0*kSystemPointerSize]: implicit receiver
    //  -- sp[1*kSystemPointerSize]: padding
    //  -- sp[2*kSystemPointerSize]: constructor function
    //  -- sp[3*kSystemPointerSize]: number of arguments
    //  -- sp[4*kSystemPointerSize]: context
    // -----------------------------------

    // Store offset of return address for deoptimizer.
    masm->isolate()->heap()->SetConstructStubInvokeDeoptPCOffset(
        masm->pc_offset());

    // Restore the context from the frame.
    __ RecordComment(
        "[  Ld(cp, MemOperand(fp, ConstructFrameConstants::kContextOffset));");
    __ Ld(cp, MemOperand(fp, ConstructFrameConstants::kContextOffset));
    __ RecordComment("]");

    // If the result is an object (in the ECMA sense), we should get rid
    // of the receiver and use the result; see ECMA-262 section 13.2.2-7
    // on page 74.
    Label use_receiver, do_throw, leave_frame;

    // If the result is undefined, we jump out to using the implicit receiver.
    __ RecordComment(
        "[  JumpIfRoot(a0, RootIndex::kUndefinedValue, &use_receiver);");
    __ JumpIfRoot(a0, RootIndex::kUndefinedValue, &use_receiver);
    __ RecordComment("]");

    // Otherwise we do a smi check and fall through to check if the return value
    // is a valid receiver.

    // If the result is a smi, it is *not* an object in the ECMA sense.
    __ RecordComment("[  JumpIfSmi(a0, &use_receiver);");
    __ JumpIfSmi(a0, &use_receiver);
    __ RecordComment("]");

    // If the type of the result (stored in its map) is less than
    // FIRST_JS_RECEIVER_TYPE, it is not an object in the ECMA sense.
    __ RecordComment("[  GetObjectType(a0, t2, t2);");
    __ GetObjectType(a0, t2, t2);
    __ RecordComment("]");
    STATIC_ASSERT(LAST_JS_RECEIVER_TYPE == LAST_TYPE);
    __ RecordComment(
        "[  Branch(&leave_frame, greater_equal, t2, "
        "Operand(FIRST_JS_RECEIVER_TYPE));");
    __ Branch(&leave_frame, greater_equal, t2, Operand(FIRST_JS_RECEIVER_TYPE));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&use_receiver);");
    __ Branch(&use_receiver);
    __ RecordComment("]");

    __ RecordComment("[  bind(&do_throw);");
    __ bind(&do_throw);
    __ RecordComment("]");
    __ RecordComment(
        "[  CallRuntime(Runtime::kThrowConstructorReturnedNonObject);");
    __ CallRuntime(Runtime::kThrowConstructorReturnedNonObject);
    __ RecordComment("]");

    // Throw away the result of the constructor invocation and use the
    // on-stack receiver as the result.
    __ RecordComment("[  bind(&use_receiver);");
    __ bind(&use_receiver);
    __ RecordComment("]");
    __ RecordComment("[  Ld(a0, MemOperand(sp, 0 * kSystemPointerSize));");
    __ Ld(a0, MemOperand(sp, 0 * kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment("[  JumpIfRoot(a0, RootIndex::kTheHoleValue, &do_throw);");
    __ JumpIfRoot(a0, RootIndex::kTheHoleValue, &do_throw);
    __ RecordComment("]");

    __ RecordComment("[  bind(&leave_frame);");
    __ bind(&leave_frame);
    __ RecordComment("]");
    // Restore smi-tagged arguments count from the frame.
    __ RecordComment(
        "[  Ld(a1, MemOperand(fp, ConstructFrameConstants::kLengthOffset));");
    __ Ld(a1, MemOperand(fp, ConstructFrameConstants::kLengthOffset));
    __ RecordComment("]");
    // Leave construct frame.
  }
  // Remove caller arguments from the stack and return.
  __ RecordComment("[  SmiScale(a4, a1, kSystemPointerSizeLog2);");
  __ SmiScale(a4, a1, kSystemPointerSizeLog2);
  __ RecordComment("]");
  __ RecordComment("[  Add64(sp, sp, a4);");
  __ Add64(sp, sp, a4);
  __ RecordComment("]");
  __ RecordComment("[  Add64(sp, sp, kSystemPointerSize);");
  __ Add64(sp, sp, kSystemPointerSize);
  __ RecordComment("]");
  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");
}

void Builtins::Generate_JSBuiltinsConstructStub(MacroAssembler* masm) {
  Generate_JSBuiltinsConstructStubHelper(masm);
}

static void GetSharedFunctionInfoBytecode(MacroAssembler* masm,
                                          Register sfi_data,
                                          Register scratch1) {
  Label done;

  __ RecordComment("[  GetObjectType(sfi_data, scratch1, scratch1);");
  __ GetObjectType(sfi_data, scratch1, scratch1);
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&done, ne, scratch1, Operand(INTERPRETER_DATA_TYPE));");
  __ Branch(&done, ne, scratch1, Operand(INTERPRETER_DATA_TYPE));
  __ RecordComment("]");
  __ LoadTaggedPointerField(
      sfi_data,
      FieldMemOperand(sfi_data, InterpreterData::kBytecodeArrayOffset));

  __ RecordComment("[  bind(&done);");
  __ bind(&done);
  __ RecordComment("]");
}

// static
void Builtins::Generate_ResumeGeneratorTrampoline(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0 : the value to pass to the generator
  //  -- a1 : the JSGeneratorObject to resume
  //  -- ra : return address
  // -----------------------------------
  __ RecordComment("[  AssertGeneratorObject(a1);");
  __ AssertGeneratorObject(a1);
  __ RecordComment("]");

  // Store input value into generator object.
  __ RecordComment(
      "[  StoreTaggedField(a0, FieldMemOperand(a1, "
      "JSGeneratorObject::kInputOrDebugPosOffset));");
  __ StoreTaggedField(
      a0, FieldMemOperand(a1, JSGeneratorObject::kInputOrDebugPosOffset));
  __ RecordComment("]");
  __ RecordWriteField(a1, JSGeneratorObject::kInputOrDebugPosOffset, a0, a3,
                      kRAHasNotBeenSaved, kDontSaveFPRegs);

  // Load suspended function and context.
  __ RecordComment(
      "[  LoadTaggedPointerField(a4, FieldMemOperand(a1, "
      "JSGeneratorObject::kFunctionOffset));");
  __ LoadTaggedPointerField(
      a4, FieldMemOperand(a1, JSGeneratorObject::kFunctionOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  LoadTaggedPointerField(cp, FieldMemOperand(a4, "
      "JSFunction::kContextOffset));");
  __ LoadTaggedPointerField(cp,
                            FieldMemOperand(a4, JSFunction::kContextOffset));
  __ RecordComment("]");

  // Flood function if we are stepping.
  Label prepare_step_in_if_stepping, prepare_step_in_suspended_generator;
  Label stepping_prepared;
  ExternalReference debug_hook =
      ExternalReference::debug_hook_on_function_call_address(masm->isolate());
  __ RecordComment("[  li(a5, debug_hook);");
  __ li(a5, debug_hook);
  __ RecordComment("]");
  __ RecordComment("[  Lb(a5, MemOperand(a5));");
  __ Lb(a5, MemOperand(a5));
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&prepare_step_in_if_stepping, ne, a5, Operand(zero_reg));");
  __ Branch(&prepare_step_in_if_stepping, ne, a5, Operand(zero_reg));
  __ RecordComment("]");

  // Flood function if we need to continue stepping in the suspended generator.
  ExternalReference debug_suspended_generator =
      ExternalReference::debug_suspended_generator_address(masm->isolate());
  __ RecordComment("[  li(a5, debug_suspended_generator);");
  __ li(a5, debug_suspended_generator);
  __ RecordComment("]");
  __ RecordComment("[  Ld(a5, MemOperand(a5));");
  __ Ld(a5, MemOperand(a5));
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&prepare_step_in_suspended_generator, eq, a1, Operand(a5));");
  __ Branch(&prepare_step_in_suspended_generator, eq, a1, Operand(a5));
  __ RecordComment("]");
  __ RecordComment("[  bind(&stepping_prepared);");
  __ bind(&stepping_prepared);
  __ RecordComment("]");

  // Check the stack for overflow. We are not trying to catch interruptions
  // (i.e. debug break and preemption) here, so check the "real stack limit".
  Label stack_overflow;
  LoadStackLimit(masm, kScratchReg, StackLimitKind::kRealStackLimit);
  __ RecordComment(
      "[  Branch(&stack_overflow, Uless, sp, Operand(kScratchReg));");
  __ Branch(&stack_overflow, Uless, sp, Operand(kScratchReg));
  __ RecordComment("]");

  // ----------- S t a t e -------------
  //  -- a1    : the JSGeneratorObject to resume
  //  -- a4    : generator function
  //  -- cp    : generator context
  //  -- ra    : return address
  // -----------------------------------

  // Push holes for arguments to generator function. Since the parser forced
  // context allocation for any variables in generators, the actual argument
  // values have already been copied into the context and these dummy values
  // will never be used.
  __ RecordComment(
      "[  LoadTaggedPointerField(a3, FieldMemOperand(a4, "
      "JSFunction::kSharedFunctionInfoOffset));");
  __ LoadTaggedPointerField(
      a3, FieldMemOperand(a4, JSFunction::kSharedFunctionInfoOffset));
  __ RecordComment("]");
  __ Lhu(a3,
         FieldMemOperand(a3, SharedFunctionInfo::kFormalParameterCountOffset));
  __ LoadTaggedPointerField(
      t1,
      FieldMemOperand(a1, JSGeneratorObject::kParametersAndRegistersOffset));
  {
    Label done_loop, loop;
    __ RecordComment("[  bind(&loop);");
    __ bind(&loop);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(a3, a3, Operand(1));");
    __ Sub64(a3, a3, Operand(1));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&done_loop, lt, a3, Operand(zero_reg));");
    __ Branch(&done_loop, lt, a3, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment(
        "[  CalcScaledAddress(kScratchReg, t1, a3, kSystemPointerSizeLog2);");
    __ CalcScaledAddress(kScratchReg, t1, a3, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment(
        "[  LoadAnyTaggedField(kScratchReg, FieldMemOperand(kScratchReg, "
        "FixedArray::kHeaderSize));");
    __ LoadAnyTaggedField(
        kScratchReg, FieldMemOperand(kScratchReg, FixedArray::kHeaderSize));
    __ RecordComment("]");
    __ RecordComment("[  Push(kScratchReg);");
    __ Push(kScratchReg);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&loop);");
    __ Branch(&loop);
    __ RecordComment("]");
    __ RecordComment("[  bind(&done_loop);");
    __ bind(&done_loop);
    __ RecordComment("]");
    // Push receiver
    __ RecordComment(
        "[  LoadTaggedPointerField(kScratchReg, FieldMemOperand(a1, "
        "JSGeneratorObject::kReceiverOffset));");
    __ LoadTaggedPointerField(
        kScratchReg, FieldMemOperand(a1, JSGeneratorObject::kReceiverOffset));
    __ RecordComment("]");
    __ RecordComment("[  Push(kScratchReg);");
    __ Push(kScratchReg);
    __ RecordComment("]");
  }

  // Underlying function needs to have bytecode available.
  if (FLAG_debug_code) {
    __ RecordComment(
        "[  LoadTaggedPointerField(a3, FieldMemOperand(a4, "
        "JSFunction::kSharedFunctionInfoOffset));");
    __ LoadTaggedPointerField(
        a3, FieldMemOperand(a4, JSFunction::kSharedFunctionInfoOffset));
    __ RecordComment("]");
    __ RecordComment(
        "[  LoadTaggedPointerField(a3, FieldMemOperand(a3, "
        "SharedFunctionInfo::kFunctionDataOffset));");
    __ LoadTaggedPointerField(
        a3, FieldMemOperand(a3, SharedFunctionInfo::kFunctionDataOffset));
    __ RecordComment("]");
    GetSharedFunctionInfoBytecode(masm, a3, a0);
    __ RecordComment("[  GetObjectType(a3, a3, a3);");
    __ GetObjectType(a3, a3, a3);
    __ RecordComment("]");
    __ Assert(eq, AbortReason::kMissingBytecodeArray, a3,
              Operand(BYTECODE_ARRAY_TYPE));
  }

  // Resume (Ignition/TurboFan) generator object.
  {
    __ RecordComment(
        "[  LoadTaggedPointerField(a0, FieldMemOperand(a4, "
        "JSFunction::kSharedFunctionInfoOffset));");
    __ LoadTaggedPointerField(
        a0, FieldMemOperand(a4, JSFunction::kSharedFunctionInfoOffset));
    __ RecordComment("]");
    __ Lhu(a0, FieldMemOperand(
                   a0, SharedFunctionInfo::kFormalParameterCountOffset));
    // We abuse new.target both to indicate that this is a resume call and to
    // pass in the generator object.  In ordinary calls, new.target is always
    // undefined because generator functions are non-constructable.
    __ RecordComment("[  Move(a3, a1);");
    __ Move(a3, a1);
    __ RecordComment("]");
    __ RecordComment("[  Move(a1, a4);");
    __ Move(a1, a4);
    __ RecordComment("]");
    static_assert(kJavaScriptCallCodeStartRegister == a2, "ABI mismatch");
    __ RecordComment(
        "[  LoadTaggedPointerField(a2, FieldMemOperand(a1, "
        "JSFunction::kCodeOffset));");
    __ LoadTaggedPointerField(a2, FieldMemOperand(a1, JSFunction::kCodeOffset));
    __ RecordComment("]");
    __ RecordComment("[  JumpCodeObject(a2);");
    __ JumpCodeObject(a2);
    __ RecordComment("]");
  }

  __ RecordComment("[  bind(&prepare_step_in_if_stepping);");
  __ bind(&prepare_step_in_if_stepping);
  __ RecordComment("]");
  {
    FrameScope scope(masm, StackFrame::INTERNAL);
    __ RecordComment("[  Push(a1, a4);");
    __ Push(a1, a4);
    __ RecordComment("]");
    // Push hole as receiver since we do not use it for stepping.
    __ RecordComment("[  PushRoot(RootIndex::kTheHoleValue);");
    __ PushRoot(RootIndex::kTheHoleValue);
    __ RecordComment("]");
    __ RecordComment("[  CallRuntime(Runtime::kDebugOnFunctionCall);");
    __ CallRuntime(Runtime::kDebugOnFunctionCall);
    __ RecordComment("]");
    __ RecordComment("[  Pop(a1);");
    __ Pop(a1);
    __ RecordComment("]");
  }
  __ RecordComment(
      "[  Ld(a4, FieldMemOperand(a1, JSGeneratorObject::kFunctionOffset));");
  __ Ld(a4, FieldMemOperand(a1, JSGeneratorObject::kFunctionOffset));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&stepping_prepared);");
  __ Branch(&stepping_prepared);
  __ RecordComment("]");

  __ RecordComment("[  bind(&prepare_step_in_suspended_generator);");
  __ bind(&prepare_step_in_suspended_generator);
  __ RecordComment("]");
  {
    FrameScope scope(masm, StackFrame::INTERNAL);
    __ RecordComment("[  Push(a1);");
    __ Push(a1);
    __ RecordComment("]");
    __ RecordComment(
        "[  CallRuntime(Runtime::kDebugPrepareStepInSuspendedGenerator);");
    __ CallRuntime(Runtime::kDebugPrepareStepInSuspendedGenerator);
    __ RecordComment("]");
    __ RecordComment("[  Pop(a1);");
    __ Pop(a1);
    __ RecordComment("]");
  }
  __ RecordComment(
      "[  Ld(a4, FieldMemOperand(a1, JSGeneratorObject::kFunctionOffset));");
  __ Ld(a4, FieldMemOperand(a1, JSGeneratorObject::kFunctionOffset));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&stepping_prepared);");
  __ Branch(&stepping_prepared);
  __ RecordComment("]");

  __ RecordComment("[  bind(&stack_overflow);");
  __ bind(&stack_overflow);
  __ RecordComment("]");
  {
    FrameScope scope(masm, StackFrame::INTERNAL);
    __ RecordComment("[  CallRuntime(Runtime::kThrowStackOverflow);");
    __ CallRuntime(Runtime::kThrowStackOverflow);
    __ RecordComment("]");
    __ RecordComment("[  break_(0xCC);  // This should be unreachable.");
    __ break_(0xCC);  // This should be unreachable.
    __ RecordComment("]");
  }
}

void Builtins::Generate_ConstructedNonConstructable(MacroAssembler* masm) {
  FrameScope scope(masm, StackFrame::INTERNAL);
  __ RecordComment("[  Push(a1);");
  __ Push(a1);
  __ RecordComment("]");
  __ RecordComment(
      "[  CallRuntime(Runtime::kThrowConstructedNonConstructable);");
  __ CallRuntime(Runtime::kThrowConstructedNonConstructable);
  __ RecordComment("]");
}

// Clobbers scratch1 and scratch2; preserves all other registers.
static void Generate_CheckStackOverflow(MacroAssembler* masm, Register argc,
                                        Register scratch1, Register scratch2) {
  // Check the stack for overflow. We are not trying to catch
  // interruptions (e.g. debug break and preemption) here, so the "real stack
  // limit" is checked.
  Label okay;
  LoadStackLimit(masm, scratch1, StackLimitKind::kRealStackLimit);
  // Make a2 the space we have left. The stack might already be overflowed
  // here which will cause r2 to become negative.
  __ RecordComment("[  Sub64(scratch1, sp, scratch1);");
  __ Sub64(scratch1, sp, scratch1);
  __ RecordComment("]");
  // Check if the arguments will overflow the stack.
  __ RecordComment("[  Sll64(scratch2, argc, kSystemPointerSizeLog2);");
  __ Sll64(scratch2, argc, kSystemPointerSizeLog2);
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&okay, gt, scratch1, Operand(scratch2));  // Signed "
      "comparison.");
  __ Branch(&okay, gt, scratch1, Operand(scratch2));  // Signed comparison.
  __ RecordComment("]");

  // Out of stack space.
  __ RecordComment("[  CallRuntime(Runtime::kThrowStackOverflow);");
  __ CallRuntime(Runtime::kThrowStackOverflow);
  __ RecordComment("]");

  __ RecordComment("[  bind(&okay);");
  __ bind(&okay);
  __ RecordComment("]");
}

namespace {

// Called with the native C calling convention. The corresponding function
// signature is either:
//
//   using JSEntryFunction = GeneratedCode<Address(
//       Address root_register_value, Address new_target, Address target,
//       Address receiver, intptr_t argc, Address** args)>;
// or
//   using JSEntryFunction = GeneratedCode<Address(
//       Address root_register_value, MicrotaskQueue* microtask_queue)>;
void Generate_JSEntryVariant(MacroAssembler* masm, StackFrame::Type type,
                             Builtins::Name entry_trampoline) {
  Label invoke, handler_entry, exit;

  {
    NoRootArrayScope no_root_array(masm);

    // TODO(plind): unify the ABI description here.
    // Registers:
    //  either
    //   a0: root register value
    //   a1: entry address
    //   a2: function
    //   a3: receiver
    //   a4: argc
    //   a5: argv
    //  or
    //   a0: root register value
    //   a1: microtask_queue

    // Save callee saved registers on the stack.
    __ RecordComment("[  MultiPush(kCalleeSaved | ra.bit());");
    __ MultiPush(kCalleeSaved | ra.bit());
    __ RecordComment("]");

    // Save callee-saved FPU registers.
    __ RecordComment("[  MultiPushFPU(kCalleeSavedFPU);");
    __ MultiPushFPU(kCalleeSavedFPU);
    __ RecordComment("]");
    // Set up the reserved register for 0.0.
    __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0);");
    __ LoadFPRImmediate(kDoubleRegZero, 0.0);
    __ RecordComment("]");

    // Initialize the root register.
    // C calling convention. The first argument is passed in a0.
    __ RecordComment("[  Move(kRootRegister, a0);");
    __ Move(kRootRegister, a0);
    __ RecordComment("]");
  }

  // a1: entry address
  // a2: function
  // a3: receiver
  // a4: argc
  // a5: argv

  // We build an EntryFrame.
  __ RecordComment(
      "[  li(s1, Operand(-1));  // Push a bad frame pointer to fail if it is "
      "used.");
  __ li(s1, Operand(-1));  // Push a bad frame pointer to fail if it is used.
  __ RecordComment("]");
  __ RecordComment("[  li(s2, Operand(StackFrame::TypeToMarker(type)));");
  __ li(s2, Operand(StackFrame::TypeToMarker(type)));
  __ RecordComment("]");
  __ RecordComment("[  li(s3, Operand(StackFrame::TypeToMarker(type)));");
  __ li(s3, Operand(StackFrame::TypeToMarker(type)));
  __ RecordComment("]");
  ExternalReference c_entry_fp = ExternalReference::Create(
      IsolateAddressId::kCEntryFPAddress, masm->isolate());
  __ RecordComment("[  li(s4, c_entry_fp);");
  __ li(s4, c_entry_fp);
  __ RecordComment("]");
  __ RecordComment("[  Ld(s4, MemOperand(s4));");
  __ Ld(s4, MemOperand(s4));
  __ RecordComment("]");
  __ RecordComment("[  Push(s1, s2, s3, s4);");
  __ Push(s1, s2, s3, s4);
  __ RecordComment("]");
  // Set up frame pointer for the frame to be pushed.
  __ RecordComment("[  Add64(fp, sp, -EntryFrameConstants::kCallerFPOffset);");
  __ Add64(fp, sp, -EntryFrameConstants::kCallerFPOffset);
  __ RecordComment("]");
  // Registers:
  //  either
  //   a1: entry address
  //   a2: function
  //   a3: receiver
  //   a4: argc
  //   a5: argv
  //  or
  //   a1: microtask_queue
  //
  // Stack:
  // caller fp          |
  // function slot      | entry frame
  // context slot       |
  // bad fp (0xFF...F)  |
  // callee saved registers + ra
  // [ O32: 4 args slots]
  // args

  // If this is the outermost JS call, set js_entry_sp value.
  Label non_outermost_js;
  ExternalReference js_entry_sp = ExternalReference::Create(
      IsolateAddressId::kJSEntrySPAddress, masm->isolate());
  __ RecordComment("[  li(s1, js_entry_sp);");
  __ li(s1, js_entry_sp);
  __ RecordComment("]");
  __ RecordComment("[  Ld(s2, MemOperand(s1));");
  __ Ld(s2, MemOperand(s1));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&non_outermost_js, ne, s2, Operand(zero_reg));");
  __ Branch(&non_outermost_js, ne, s2, Operand(zero_reg));
  __ RecordComment("]");
  __ RecordComment("[  Sd(fp, MemOperand(s1));");
  __ Sd(fp, MemOperand(s1));
  __ RecordComment("]");
  __ RecordComment("[  li(s3, Operand(StackFrame::OUTERMOST_JSENTRY_FRAME));");
  __ li(s3, Operand(StackFrame::OUTERMOST_JSENTRY_FRAME));
  __ RecordComment("]");
  Label cont;
  __ RecordComment("[  Branch(&cont);");
  __ Branch(&cont);
  __ RecordComment("]");
  __ RecordComment("[  bind(&non_outermost_js);");
  __ bind(&non_outermost_js);
  __ RecordComment("]");
  __ RecordComment("[  li(s3, Operand(StackFrame::INNER_JSENTRY_FRAME));");
  __ li(s3, Operand(StackFrame::INNER_JSENTRY_FRAME));
  __ RecordComment("]");
  __ RecordComment("[  bind(&cont);");
  __ bind(&cont);
  __ RecordComment("]");
  __ RecordComment("[  push(s3);");
  __ push(s3);
  __ RecordComment("]");

  // Jump to a faked try block that does the invoke, with a faked catch
  // block that sets the pending exception.
  __ RecordComment("[  Branch(&invoke);");
  __ Branch(&invoke);
  __ RecordComment("]");
  __ RecordComment("[  bind(&handler_entry);");
  __ bind(&handler_entry);
  __ RecordComment("]");

  // Store the current pc as the handler offset. It's used later to create the
  // handler table.
  masm->isolate()->builtins()->SetJSEntryHandlerOffset(handler_entry.pos());

  // Caught exception: Store result (exception) in the pending exception
  // field in the JSEnv and return a failure sentinel.  Coming in here the
  // fp will be invalid because the PushStackHandler below sets it to 0 to
  // signal the existence of the JSEntry frame.
  __ li(s1, ExternalReference::Create(
                IsolateAddressId::kPendingExceptionAddress, masm->isolate()));
  __ RecordComment(
      "[  Sd(a0, MemOperand(s1));  // We come back from 'invoke'. result is in "
      "a0.");
  __ Sd(a0, MemOperand(s1));  // We come back from 'invoke'. result is in a0.
  __ RecordComment("]");
  __ RecordComment("[  LoadRoot(a0, RootIndex::kException);");
  __ LoadRoot(a0, RootIndex::kException);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&exit);");
  __ Branch(&exit);
  __ RecordComment("]");

  // Invoke: Link this frame into the handler chain.
  __ RecordComment("[  bind(&invoke);");
  __ bind(&invoke);
  __ RecordComment("]");
  __ RecordComment("[  PushStackHandler();");
  __ PushStackHandler();
  __ RecordComment("]");
  // If an exception not caught by another handler occurs, this handler
  // returns control to the code after the bal(&invoke) above, which
  // restores all kCalleeSaved registers (including cp and fp) to their
  // saved values before returning a failure to C.
  //
  // Registers:
  //  either
  //   a0: root register value
  //   a1: entry address
  //   a2: function
  //   a3: receiver
  //   a4: argc
  //   a5: argv
  //  or
  //   a0: root register value
  //   a1: microtask_queue
  //
  // Stack:
  // handler frame
  // entry frame
  // callee saved registers + ra
  // [ O32: 4 args slots]
  // args
  //
  // Invoke the function by calling through JS entry trampoline builtin and
  // pop the faked function when we return.

  Handle<Code> trampoline_code =
      masm->isolate()->builtins()->builtin_handle(entry_trampoline);
  __ RecordComment("[  Call(trampoline_code, RelocInfo::CODE_TARGET);");
  __ Call(trampoline_code, RelocInfo::CODE_TARGET);
  __ RecordComment("]");

  // Unlink this frame from the handler chain.
  __ RecordComment("[  PopStackHandler();");
  __ PopStackHandler();
  __ RecordComment("]");

  __ RecordComment("[  bind(&exit);  // a0 holds result");
  __ bind(&exit);  // a0 holds result
  __ RecordComment("]");
  // Check if the current stack frame is marked as the outermost JS frame.
  Label non_outermost_js_2;
  __ RecordComment("[  pop(a5);");
  __ pop(a5);
  __ RecordComment("]");
  __ Branch(&non_outermost_js_2, ne, a5,
            Operand(StackFrame::OUTERMOST_JSENTRY_FRAME));
  __ RecordComment("[  li(a5, js_entry_sp);");
  __ li(a5, js_entry_sp);
  __ RecordComment("]");
  __ RecordComment("[  Sd(zero_reg, MemOperand(a5));");
  __ Sd(zero_reg, MemOperand(a5));
  __ RecordComment("]");
  __ RecordComment("[  bind(&non_outermost_js_2);");
  __ bind(&non_outermost_js_2);
  __ RecordComment("]");

  // Restore the top frame descriptors from the stack.
  __ RecordComment("[  pop(a5);");
  __ pop(a5);
  __ RecordComment("]");
  __ li(a4, ExternalReference::Create(IsolateAddressId::kCEntryFPAddress,
                                      masm->isolate()));
  __ RecordComment("[  Sd(a5, MemOperand(a4));");
  __ Sd(a5, MemOperand(a4));
  __ RecordComment("]");

  // Reset the stack to the callee saved registers.
  __ RecordComment("[  Add64(sp, sp, -EntryFrameConstants::kCallerFPOffset);");
  __ Add64(sp, sp, -EntryFrameConstants::kCallerFPOffset);
  __ RecordComment("]");

  // Restore callee-saved fpu registers.
  __ RecordComment("[  MultiPopFPU(kCalleeSavedFPU);");
  __ MultiPopFPU(kCalleeSavedFPU);
  __ RecordComment("]");

  // Restore callee saved registers from the stack.
  __ RecordComment("[  MultiPop(kCalleeSaved | ra.bit());");
  __ MultiPop(kCalleeSaved | ra.bit());
  __ RecordComment("]");
  // Return.
  __ RecordComment("[  Jump(ra);");
  __ Jump(ra);
  __ RecordComment("]");
}

}  // namespace

void Builtins::Generate_JSEntry(MacroAssembler* masm) {
  Generate_JSEntryVariant(masm, StackFrame::ENTRY,
                          Builtins::kJSEntryTrampoline);
}

void Builtins::Generate_JSConstructEntry(MacroAssembler* masm) {
  Generate_JSEntryVariant(masm, StackFrame::CONSTRUCT_ENTRY,
                          Builtins::kJSConstructEntryTrampoline);
}

void Builtins::Generate_JSRunMicrotasksEntry(MacroAssembler* masm) {
  Generate_JSEntryVariant(masm, StackFrame::ENTRY,
                          Builtins::kRunMicrotasksTrampoline);
}

static void Generate_JSEntryTrampolineHelper(MacroAssembler* masm,
                                             bool is_construct) {
  // ----------- S t a t e -------------
  //  -- a1: new.target
  //  -- a2: function
  //  -- a3: receiver_pointer
  //  -- a4: argc
  //  -- a5: argv
  // -----------------------------------

  // Enter an internal frame.
  {
    FrameScope scope(masm, StackFrame::INTERNAL);

    // Setup the context (we need to use the caller context from the isolate).
    ExternalReference context_address = ExternalReference::Create(
        IsolateAddressId::kContextAddress, masm->isolate());
    __ RecordComment("[  li(cp, context_address);");
    __ li(cp, context_address);
    __ RecordComment("]");
    __ RecordComment("[  Ld(cp, MemOperand(cp));");
    __ Ld(cp, MemOperand(cp));
    __ RecordComment("]");

    // Push the function onto the stack.
    __ RecordComment("[  Push(a2);");
    __ Push(a2);
    __ RecordComment("]");

    // Check if we have enough stack space to push all arguments.
    __ RecordComment("[  Add64(a6, a4, 1);");
    __ Add64(a6, a4, 1);
    __ RecordComment("]");
    Generate_CheckStackOverflow(masm, a6, a0, s2);

    // Copy arguments to the stack in a loop.
    // a4: argc
    // a5: argv, i.e. points to first arg
    Label loop, entry;
    __ RecordComment(
        "[  CalcScaledAddress(s1, a5, a4, kSystemPointerSizeLog2);");
    __ CalcScaledAddress(s1, a5, a4, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&entry);");
    __ Branch(&entry);
    __ RecordComment("]");
    // s1 points past last arg.
    __ RecordComment("[  bind(&loop);");
    __ bind(&loop);
    __ RecordComment("]");
    __ RecordComment("[  Add64(s1, s1, -kSystemPointerSize);");
    __ Add64(s1, s1, -kSystemPointerSize);
    __ RecordComment("]");
    __ RecordComment("[  Ld(s2, MemOperand(s1));  // Read next parameter.");
    __ Ld(s2, MemOperand(s1));  // Read next parameter.
    __ RecordComment("]");
    __ RecordComment("[  Ld(s2, MemOperand(s2));  // Dereference handle.");
    __ Ld(s2, MemOperand(s2));  // Dereference handle.
    __ RecordComment("]");
    __ RecordComment("[  push(s2);                // Push parameter.");
    __ push(s2);  // Push parameter.
    __ RecordComment("]");
    __ RecordComment("[  bind(&entry);");
    __ bind(&entry);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&loop, ne, a5, Operand(s1));");
    __ Branch(&loop, ne, a5, Operand(s1));
    __ RecordComment("]");

    // Push the receive.
    __ RecordComment("[  Push(a3);");
    __ Push(a3);
    __ RecordComment("]");

    // a0: argc
    // a1: function
    // a3: new.target
    __ RecordComment("[  Move(a3, a1);");
    __ Move(a3, a1);
    __ RecordComment("]");
    __ RecordComment("[  Move(a1, a2);");
    __ Move(a1, a2);
    __ RecordComment("]");
    __ RecordComment("[  Move(a0, a4);");
    __ Move(a0, a4);
    __ RecordComment("]");

    // Initialize all JavaScript callee-saved registers, since they will be seen
    // by the garbage collector as part of handlers.
    __ RecordComment("[  LoadRoot(a4, RootIndex::kUndefinedValue);");
    __ LoadRoot(a4, RootIndex::kUndefinedValue);
    __ RecordComment("]");
    __ RecordComment("[  Move(a5, a4);");
    __ Move(a5, a4);
    __ RecordComment("]");
    __ RecordComment("[  Move(s1, a4);");
    __ Move(s1, a4);
    __ RecordComment("]");
    __ RecordComment("[  Move(s2, a4);");
    __ Move(s2, a4);
    __ RecordComment("]");
    __ RecordComment("[  Move(s3, a4);");
    __ Move(s3, a4);
    __ RecordComment("]");
    __ RecordComment("[  Move(s4, a4);");
    __ Move(s4, a4);
    __ RecordComment("]");
    __ RecordComment("[  Move(s5, a4);");
    __ Move(s5, a4);
    __ RecordComment("]");
    // s6 holds the root address. Do not clobber.
    // s7 is cp. Do not init.

    // Invoke the code.
    Handle<Code> builtin = is_construct
                               ? BUILTIN_CODE(masm->isolate(), Construct)
                               : masm->isolate()->builtins()->Call();
    __ RecordComment("[  Call(builtin, RelocInfo::CODE_TARGET);");
    __ Call(builtin, RelocInfo::CODE_TARGET);
    __ RecordComment("]");

    // Leave internal frame.
  }
  __ RecordComment("[  Jump(ra);");
  __ Jump(ra);
  __ RecordComment("]");
}

void Builtins::Generate_JSEntryTrampoline(MacroAssembler* masm) {
  Generate_JSEntryTrampolineHelper(masm, false);
}

void Builtins::Generate_JSConstructEntryTrampoline(MacroAssembler* masm) {
  Generate_JSEntryTrampolineHelper(masm, true);
}

void Builtins::Generate_RunMicrotasksTrampoline(MacroAssembler* masm) {
  // a1: microtask_queue
  __ RecordComment(
      "[  Move(RunMicrotasksDescriptor::MicrotaskQueueRegister(), a1);");
  __ Move(RunMicrotasksDescriptor::MicrotaskQueueRegister(), a1);
  __ RecordComment("]");
  __ RecordComment(
      "[  Jump(BUILTIN_CODE(masm->isolate(), RunMicrotasks), "
      "RelocInfo::CODE_TARGET);");
  __ Jump(BUILTIN_CODE(masm->isolate(), RunMicrotasks), RelocInfo::CODE_TARGET);
  __ RecordComment("]");
}

static void ReplaceClosureCodeWithOptimizedCode(MacroAssembler* masm,
                                                Register optimized_code,
                                                Register closure,
                                                Register scratch1,
                                                Register scratch2) {
  // Store code entry in the closure.
  __ RecordComment(
      "[  StoreTaggedField(optimized_code, FieldMemOperand(closure, "
      "JSFunction::kCodeOffset));");
  __ StoreTaggedField(optimized_code,
                      FieldMemOperand(closure, JSFunction::kCodeOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Move(scratch1, optimized_code);  // Write barrier clobbers scratch1 "
      "below.");
  __ Move(scratch1, optimized_code);  // Write barrier clobbers scratch1 below.
  __ RecordComment("]");
  __ RecordWriteField(closure, JSFunction::kCodeOffset, scratch1, scratch2,
                      kRAHasNotBeenSaved, kDontSaveFPRegs, OMIT_REMEMBERED_SET,
                      OMIT_SMI_CHECK);
}

static void LeaveInterpreterFrame(MacroAssembler* masm, Register scratch) {
  Register args_count = scratch;

  // Get the arguments + receiver count.
  __ Ld(args_count,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeArrayFromFp));
  __ RecordComment(
      "[  Lw(t0, FieldMemOperand(args_count, "
      "BytecodeArray::kParameterSizeOffset));");
  __ Lw(t0, FieldMemOperand(args_count, BytecodeArray::kParameterSizeOffset));
  __ RecordComment("]");

  // Leave the frame (also dropping the register file).
  __ RecordComment("[  LeaveFrame(StackFrame::INTERPRETED);");
  __ LeaveFrame(StackFrame::INTERPRETED);
  __ RecordComment("]");

  // Drop receiver + arguments.
  __ RecordComment("[  Add64(sp, sp, args_count);");
  __ Add64(sp, sp, args_count);
  __ RecordComment("]");
}

// Tail-call |function_id| if |smi_entry| == |marker|
static void TailCallRuntimeIfMarkerEquals(MacroAssembler* masm,
                                          Register smi_entry,
                                          OptimizationMarker marker,
                                          Runtime::FunctionId function_id) {
  Label no_match;
  __ RecordComment(
      "[  Branch(&no_match, ne, smi_entry, Operand(Smi::FromEnum(marker)));");
  __ Branch(&no_match, ne, smi_entry, Operand(Smi::FromEnum(marker)));
  __ RecordComment("]");
  GenerateTailCallToReturnedCode(masm, function_id);
  __ RecordComment("[  bind(&no_match);");
  __ bind(&no_match);
  __ RecordComment("]");
}

static void TailCallOptimizedCodeSlot(MacroAssembler* masm,
                                      Register optimized_code_entry,
                                      Register scratch1, Register scratch2) {
  // ----------- S t a t e -------------
  //  -- a0 : actual argument count
  //  -- a3 : new target (preserved for callee if needed, and caller)
  //  -- a1 : target function (preserved for callee if needed, and caller)
  // -----------------------------------
  DCHECK(!AreAliased(optimized_code_entry, a1, a3, scratch1, scratch2));

  Register closure = a1;

  // Check if the optimized code is marked for deopt. If it is, call the
  // runtime to clear it.
  Label found_deoptimized_code;
  __ LoadTaggedPointerField(
      a5,
      FieldMemOperand(optimized_code_entry, Code::kCodeDataContainerOffset));
  __ RecordComment(
      "[  Lw(a5, FieldMemOperand(a5, "
      "CodeDataContainer::kKindSpecificFlagsOffset));");
  __ Lw(a5, FieldMemOperand(a5, CodeDataContainer::kKindSpecificFlagsOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  And(a5, a5, Operand(1 << Code::kMarkedForDeoptimizationBit));");
  __ And(a5, a5, Operand(1 << Code::kMarkedForDeoptimizationBit));
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&found_deoptimized_code, ne, a5, Operand(zero_reg));");
  __ Branch(&found_deoptimized_code, ne, a5, Operand(zero_reg));
  __ RecordComment("]");

  // Optimized code is good, get it into the closure and link the closure into
  // the optimized functions list, then tail call the optimized code.
  // The feedback vector is no longer used, so re-use it as a scratch
  // register.
  ReplaceClosureCodeWithOptimizedCode(masm, optimized_code_entry, closure,
                                      scratch1, scratch2);

  static_assert(kJavaScriptCallCodeStartRegister == a2, "ABI mismatch");
  __ RecordComment("[  LoadCodeObjectEntry(a2, optimized_code_entry);");
  __ LoadCodeObjectEntry(a2, optimized_code_entry);
  __ RecordComment("]");
  __ RecordComment("[  Jump(a2);");
  __ Jump(a2);
  __ RecordComment("]");

  // Optimized code slot contains deoptimized code, evict it and re-enter the
  // closure's code.
  __ RecordComment("[  bind(&found_deoptimized_code);");
  __ bind(&found_deoptimized_code);
  __ RecordComment("]");
  GenerateTailCallToReturnedCode(masm, Runtime::kEvictOptimizedCodeSlot);
}

static void MaybeOptimizeCode(MacroAssembler* masm, Register feedback_vector,
                              Register optimization_marker) {
  // ----------- S t a t e -------------
  //  -- a0 : actual argument count
  //  -- a3 : new target (preserved for callee if needed, and caller)
  //  -- a1 : target function (preserved for callee if needed, and caller)
  //  -- feedback vector (preserved for caller if needed)
  //  -- optimization_marker : a Smi containing a non-zero optimization marker.
  // -----------------------------------
  DCHECK(!AreAliased(feedback_vector, a1, a3, optimization_marker));

  // TODO(v8:8394): The logging of first execution will break if
  // feedback vectors are not allocated. We need to find a different way of
  // logging these events if required.
  TailCallRuntimeIfMarkerEquals(masm, optimization_marker,
                                OptimizationMarker::kLogFirstExecution,
                                Runtime::kFunctionFirstExecution);
  TailCallRuntimeIfMarkerEquals(masm, optimization_marker,
                                OptimizationMarker::kCompileOptimized,
                                Runtime::kCompileOptimized_NotConcurrent);
  TailCallRuntimeIfMarkerEquals(masm, optimization_marker,
                                OptimizationMarker::kCompileOptimizedConcurrent,
                                Runtime::kCompileOptimized_Concurrent);

  // Otherwise, the marker is InOptimizationQueue, so fall through hoping
  // that an interrupt will eventually update the slot with optimized code.
  if (FLAG_debug_code) {
    __ Assert(eq, AbortReason::kExpectedOptimizationSentinel,
              optimization_marker,
              Operand(Smi::FromEnum(OptimizationMarker::kInOptimizationQueue)));
  }
}

// Advance the current bytecode offset. This simulates what all bytecode
// handlers do upon completion of the underlying operation. Will bail out to a
// label if the bytecode (without prefix) is a return bytecode. Will not advance
// the bytecode offset if the current bytecode is a JumpLoop, instead just
// re-executing the JumpLoop to jump to the correct bytecode.
static void AdvanceBytecodeOffsetOrReturn(MacroAssembler* masm,
                                          Register bytecode_array,
                                          Register bytecode_offset,
                                          Register bytecode, Register scratch1,
                                          Register scratch2, Register scratch3,
                                          Label* if_return) {
  Register bytecode_size_table = scratch1;

  // The bytecode offset value will be increased by one in wide and extra wide
  // cases. In the case of having a wide or extra wide JumpLoop bytecode, we
  // will restore the original bytecode. In order to simplify the code, we have
  // a backup of it.
  Register original_bytecode_offset = scratch3;
  DCHECK(!AreAliased(bytecode_array, bytecode_offset, bytecode,
                     bytecode_size_table, original_bytecode_offset));
  __ RecordComment("[  Move(original_bytecode_offset, bytecode_offset);");
  __ Move(original_bytecode_offset, bytecode_offset);
  __ RecordComment("]");
  __ RecordComment(
      "[  li(bytecode_size_table, "
      "ExternalReference::bytecode_size_table_address());");
  __ li(bytecode_size_table, ExternalReference::bytecode_size_table_address());
  __ RecordComment("]");

  // Check if the bytecode is a Wide or ExtraWide prefix bytecode.
  Label process_bytecode, extra_wide;
  STATIC_ASSERT(0 == static_cast<int>(interpreter::Bytecode::kWide));
  STATIC_ASSERT(1 == static_cast<int>(interpreter::Bytecode::kExtraWide));
  STATIC_ASSERT(2 == static_cast<int>(interpreter::Bytecode::kDebugBreakWide));
  STATIC_ASSERT(3 ==
                static_cast<int>(interpreter::Bytecode::kDebugBreakExtraWide));
  __ RecordComment(
      "[  Branch(&process_bytecode, Ugreater, bytecode, Operand(3));");
  __ Branch(&process_bytecode, Ugreater, bytecode, Operand(3));
  __ RecordComment("]");
  __ RecordComment("[  And(scratch2, bytecode, Operand(1));");
  __ And(scratch2, bytecode, Operand(1));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&extra_wide, ne, scratch2, Operand(zero_reg));");
  __ Branch(&extra_wide, ne, scratch2, Operand(zero_reg));
  __ RecordComment("]");

  // Load the next bytecode and update table to the wide scaled table.
  __ RecordComment("[  Add64(bytecode_offset, bytecode_offset, Operand(1));");
  __ Add64(bytecode_offset, bytecode_offset, Operand(1));
  __ RecordComment("]");
  __ RecordComment("[  Add64(scratch2, bytecode_array, bytecode_offset);");
  __ Add64(scratch2, bytecode_array, bytecode_offset);
  __ RecordComment("]");
  __ RecordComment("[  Lbu(bytecode, MemOperand(scratch2));");
  __ Lbu(bytecode, MemOperand(scratch2));
  __ RecordComment("]");
  __ Add64(bytecode_size_table, bytecode_size_table,
           Operand(kIntSize * interpreter::Bytecodes::kBytecodeCount));
  __ RecordComment("[  Branch(&process_bytecode);");
  __ Branch(&process_bytecode);
  __ RecordComment("]");

  __ RecordComment("[  bind(&extra_wide);");
  __ bind(&extra_wide);
  __ RecordComment("]");
  // Load the next bytecode and update table to the extra wide scaled table.
  __ RecordComment("[  Add64(bytecode_offset, bytecode_offset, Operand(1));");
  __ Add64(bytecode_offset, bytecode_offset, Operand(1));
  __ RecordComment("]");
  __ RecordComment("[  Add64(scratch2, bytecode_array, bytecode_offset);");
  __ Add64(scratch2, bytecode_array, bytecode_offset);
  __ RecordComment("]");
  __ RecordComment("[  Lbu(bytecode, MemOperand(scratch2));");
  __ Lbu(bytecode, MemOperand(scratch2));
  __ RecordComment("]");
  __ Add64(bytecode_size_table, bytecode_size_table,
           Operand(2 * kIntSize * interpreter::Bytecodes::kBytecodeCount));

  __ RecordComment("[  bind(&process_bytecode);");
  __ bind(&process_bytecode);
  __ RecordComment("]");

// Bailout to the return label if this is a return bytecode.
#define JUMP_IF_EQUAL(NAME)          \
  __ Branch(if_return, eq, bytecode, \
            Operand(static_cast<int>(interpreter::Bytecode::k##NAME)));
  RETURN_BYTECODE_LIST(JUMP_IF_EQUAL)
#undef JUMP_IF_EQUAL

  // If this is a JumpLoop, re-execute it to perform the jump to the beginning
  // of the loop.
  Label end, not_jump_loop;
  __ Branch(&not_jump_loop, ne, bytecode,
            Operand(static_cast<int>(interpreter::Bytecode::kJumpLoop)));
  // We need to restore the original bytecode_offset since we might have
  // increased it to skip the wide / extra-wide prefix bytecode.
  __ RecordComment("[  Move(bytecode_offset, original_bytecode_offset);");
  __ Move(bytecode_offset, original_bytecode_offset);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&end);");
  __ Branch(&end);
  __ RecordComment("]");

  __ RecordComment("[  bind(&not_jump_loop);");
  __ bind(&not_jump_loop);
  __ RecordComment("]");
  // Otherwise, load the size of the current bytecode and advance the offset.
  __ RecordComment(
      "[  CalcScaledAddress(scratch2, bytecode_size_table, bytecode, 2);");
  __ CalcScaledAddress(scratch2, bytecode_size_table, bytecode, 2);
  __ RecordComment("]");
  __ RecordComment("[  Lw(scratch2, MemOperand(scratch2));");
  __ Lw(scratch2, MemOperand(scratch2));
  __ RecordComment("]");
  __ RecordComment("[  Add64(bytecode_offset, bytecode_offset, scratch2);");
  __ Add64(bytecode_offset, bytecode_offset, scratch2);
  __ RecordComment("]");

  __ RecordComment("[  bind(&end);");
  __ bind(&end);
  __ RecordComment("]");
}

// Generate code for entering a JS function with the interpreter.
// On entry to the function the receiver and arguments have been pushed on the
// stack left to right.
//
// The live registers are:
//   o a0 : actual argument count (not including the receiver)
//   o a1: the JS function object being called.
//   o a3: the incoming new target or generator object
//   o cp: our context
//   o fp: the caller's frame pointer
//   o sp: stack pointer
//   o ra: return address
//
// The function builds an interpreter frame.  See InterpreterFrameConstants in
// frames.h for its layout.
void Builtins::Generate_InterpreterEntryTrampoline(MacroAssembler* masm) {
  Register closure = a1;
  Register feedback_vector = a2;

  // Get the bytecode array from the function object and load it into
  // kInterpreterBytecodeArrayRegister.
  __ LoadTaggedPointerField(
      kScratchReg,
      FieldMemOperand(closure, JSFunction::kSharedFunctionInfoOffset));
  __ LoadTaggedPointerField(
      kInterpreterBytecodeArrayRegister,
      FieldMemOperand(kScratchReg, SharedFunctionInfo::kFunctionDataOffset));
  GetSharedFunctionInfoBytecode(masm, kInterpreterBytecodeArrayRegister,
                                kScratchReg);

  // The bytecode array could have been flushed from the shared function info,
  // if so, call into CompileLazy.
  Label compile_lazy;
  __ RecordComment(
      "[  GetObjectType(kInterpreterBytecodeArrayRegister, kScratchReg, "
      "kScratchReg);");
  __ GetObjectType(kInterpreterBytecodeArrayRegister, kScratchReg, kScratchReg);
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&compile_lazy, ne, kScratchReg, "
      "Operand(BYTECODE_ARRAY_TYPE));");
  __ Branch(&compile_lazy, ne, kScratchReg, Operand(BYTECODE_ARRAY_TYPE));
  __ RecordComment("]");

  // Load the feedback vector from the closure.
  __ LoadTaggedPointerField(
      feedback_vector,
      FieldMemOperand(closure, JSFunction::kFeedbackCellOffset));
  __ RecordComment(
      "[  LoadTaggedPointerField(feedback_vector, "
      "FieldMemOperand(feedback_vector, Cell::kValueOffset));");
  __ LoadTaggedPointerField(
      feedback_vector, FieldMemOperand(feedback_vector, Cell::kValueOffset));
  __ RecordComment("]");

  Label push_stack_frame;
  // Check if feedback vector is valid. If valid, check for optimized code
  // and update invocation count. Otherwise, setup the stack frame.
  __ RecordComment(
      "[  LoadTaggedPointerField(a4, FieldMemOperand(feedback_vector, "
      "HeapObject::kMapOffset));");
  __ LoadTaggedPointerField(
      a4, FieldMemOperand(feedback_vector, HeapObject::kMapOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Lhu(a4, FieldMemOperand(a4, Map::kInstanceTypeOffset));");
  __ Lhu(a4, FieldMemOperand(a4, Map::kInstanceTypeOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&push_stack_frame, ne, a4, Operand(FEEDBACK_VECTOR_TYPE));");
  __ Branch(&push_stack_frame, ne, a4, Operand(FEEDBACK_VECTOR_TYPE));
  __ RecordComment("]");

  // Read off the optimized code slot in the feedback vector, and if there
  // is optimized code or an optimization marker, call that instead.
  Register optimized_code_entry = a4;
  __ Ld(optimized_code_entry,
        FieldMemOperand(feedback_vector,
                        FeedbackVector::kOptimizedCodeWeakOrSmiOffset));

  // Check if the optimized code slot is not empty.
  Label optimized_code_slot_not_empty;

  __ Branch(&optimized_code_slot_not_empty, ne, optimized_code_entry,
            Operand(Smi::FromEnum(OptimizationMarker::kNone)));

  Label not_optimized;
  __ RecordComment("[  bind(&not_optimized);");
  __ bind(&not_optimized);
  __ RecordComment("]");

  // Increment invocation count for the function.
  __ Lw(a4, FieldMemOperand(feedback_vector,
                            FeedbackVector::kInvocationCountOffset));
  __ RecordComment("[  Add32(a4, a4, Operand(1));");
  __ Add32(a4, a4, Operand(1));
  __ RecordComment("]");
  __ Sw(a4, FieldMemOperand(feedback_vector,
                            FeedbackVector::kInvocationCountOffset));

  // Open a frame scope to indicate that there is a frame on the stack.  The
  // MANUAL indicates that the scope shouldn't actually generate code to set up
  // the frame (that is done below).
  __ RecordComment("[  bind(&push_stack_frame);");
  __ bind(&push_stack_frame);
  __ RecordComment("]");
  FrameScope frame_scope(masm, StackFrame::MANUAL);
  __ RecordComment("[  PushStandardFrame(closure);");
  __ PushStandardFrame(closure);
  __ RecordComment("]");

  // Reset code age and the OSR arming. The OSR field and BytecodeAgeOffset are
  // 8-bit fields next to each other, so we could just optimize by writing a
  // 16-bit. These static asserts guard our assumption is valid.
  STATIC_ASSERT(BytecodeArray::kBytecodeAgeOffset ==
                BytecodeArray::kOsrNestingLevelOffset + kCharSize);
  STATIC_ASSERT(BytecodeArray::kNoAgeBytecodeAge == 0);
  __ Sh(zero_reg, FieldMemOperand(kInterpreterBytecodeArrayRegister,
                                  BytecodeArray::kOsrNestingLevelOffset));

  // Load initial bytecode offset.
  __ li(kInterpreterBytecodeOffsetRegister,
        Operand(BytecodeArray::kHeaderSize - kHeapObjectTag));

  // Push bytecode array and Smi tagged bytecode array offset.
  __ RecordComment("[  SmiTag(a4, kInterpreterBytecodeOffsetRegister);");
  __ SmiTag(a4, kInterpreterBytecodeOffsetRegister);
  __ RecordComment("]");
  __ RecordComment("[  Push(kInterpreterBytecodeArrayRegister, a4);");
  __ Push(kInterpreterBytecodeArrayRegister, a4);
  __ RecordComment("]");

  // Allocate the local and temporary register file on the stack.
  Label stack_overflow;
  {
    // Load frame size (word) from the BytecodeArray object.
    __ Lw(a4, FieldMemOperand(kInterpreterBytecodeArrayRegister,
                              BytecodeArray::kFrameSizeOffset));

    // Do a stack check to ensure we don't go over the limit.
    __ RecordComment("[  Sub64(a5, sp, Operand(a4));");
    __ Sub64(a5, sp, Operand(a4));
    __ RecordComment("]");
    LoadStackLimit(masm, a2, StackLimitKind::kRealStackLimit);
    __ RecordComment("[  Branch(&stack_overflow, Uless, a5, Operand(a2));");
    __ Branch(&stack_overflow, Uless, a5, Operand(a2));
    __ RecordComment("]");

    // If ok, push undefined as the initial value for all register file entries.
    Label loop_header;
    Label loop_check;
    __ RecordComment("[  LoadRoot(a5, RootIndex::kUndefinedValue);");
    __ LoadRoot(a5, RootIndex::kUndefinedValue);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&loop_check);");
    __ Branch(&loop_check);
    __ RecordComment("]");
    __ RecordComment("[  bind(&loop_header);");
    __ bind(&loop_header);
    __ RecordComment("]");
    // TODO(rmcilroy): Consider doing more than one push per loop iteration.
    __ RecordComment("[  push(a5);");
    __ push(a5);
    __ RecordComment("]");
    // Continue loop if not done.
    __ RecordComment("[  bind(&loop_check);");
    __ bind(&loop_check);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(a4, a4, Operand(kSystemPointerSize));");
    __ Sub64(a4, a4, Operand(kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&loop_header, ge, a4, Operand(zero_reg));");
    __ Branch(&loop_header, ge, a4, Operand(zero_reg));
    __ RecordComment("]");
  }

  // If the bytecode array has a valid incoming new target or generator object
  // register, initialize it with incoming value which was passed in r3.
  Label no_incoming_new_target_or_generator_register;
  __ Lw(a5, FieldMemOperand(
                kInterpreterBytecodeArrayRegister,
                BytecodeArray::kIncomingNewTargetOrGeneratorRegisterOffset));
  __ Branch(&no_incoming_new_target_or_generator_register, eq, a5,
            Operand(zero_reg));
  __ RecordComment("[  CalcScaledAddress(a5, fp, a5, kSystemPointerSizeLog2);");
  __ CalcScaledAddress(a5, fp, a5, kSystemPointerSizeLog2);
  __ RecordComment("]");
  __ RecordComment("[  Sd(a3, MemOperand(a5));");
  __ Sd(a3, MemOperand(a5));
  __ RecordComment("]");
  __ RecordComment("[  bind(&no_incoming_new_target_or_generator_register);");
  __ bind(&no_incoming_new_target_or_generator_register);
  __ RecordComment("]");

  // Perform interrupt stack check.
  // TODO(solanes): Merge with the real stack limit check above.
  Label stack_check_interrupt, after_stack_check_interrupt;
  LoadStackLimit(masm, a5, StackLimitKind::kInterruptStackLimit);
  __ RecordComment(
      "[  Branch(&stack_check_interrupt, Uless, sp, Operand(a5));");
  __ Branch(&stack_check_interrupt, Uless, sp, Operand(a5));
  __ RecordComment("]");
  __ RecordComment("[  bind(&after_stack_check_interrupt);");
  __ bind(&after_stack_check_interrupt);
  __ RecordComment("]");

  // Load accumulator as undefined.
  __ RecordComment(
      "[  LoadRoot(kInterpreterAccumulatorRegister, "
      "RootIndex::kUndefinedValue);");
  __ LoadRoot(kInterpreterAccumulatorRegister, RootIndex::kUndefinedValue);
  __ RecordComment("]");

  // Load the dispatch table into a register and dispatch to the bytecode
  // handler at the current bytecode offset.
  Label do_dispatch;
  __ RecordComment("[  bind(&do_dispatch);");
  __ bind(&do_dispatch);
  __ RecordComment("]");
  __ li(kInterpreterDispatchTableRegister,
        ExternalReference::interpreter_dispatch_table_address(masm->isolate()));
  __ Add64(a1, kInterpreterBytecodeArrayRegister,
           kInterpreterBytecodeOffsetRegister);
  __ RecordComment("[  Lbu(a7, MemOperand(a1));");
  __ Lbu(a7, MemOperand(a1));
  __ RecordComment("]");
  __ CalcScaledAddress(kScratchReg, kInterpreterDispatchTableRegister, a7,
                       kSystemPointerSizeLog2);
  __ RecordComment(
      "[  Ld(kJavaScriptCallCodeStartRegister, MemOperand(kScratchReg));");
  __ Ld(kJavaScriptCallCodeStartRegister, MemOperand(kScratchReg));
  __ RecordComment("]");
  __ RecordComment("[  Call(kJavaScriptCallCodeStartRegister);");
  __ Call(kJavaScriptCallCodeStartRegister);
  __ RecordComment("]");
  masm->isolate()->heap()->SetInterpreterEntryReturnPCOffset(masm->pc_offset());

  // Any returns to the entry trampoline are either due to the return bytecode
  // or the interpreter tail calling a builtin and then a dispatch.

  // Get bytecode array and bytecode offset from the stack frame.
  __ Ld(kInterpreterBytecodeArrayRegister,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeArrayFromFp));
  __ Ld(kInterpreterBytecodeOffsetRegister,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeOffsetFromFp));
  __ RecordComment("[  SmiUntag(kInterpreterBytecodeOffsetRegister);");
  __ SmiUntag(kInterpreterBytecodeOffsetRegister);
  __ RecordComment("]");

  // Either return, or advance to the next bytecode and dispatch.
  Label do_return;
  __ Add64(a1, kInterpreterBytecodeArrayRegister,
           kInterpreterBytecodeOffsetRegister);
  __ RecordComment("[  Lbu(a1, MemOperand(a1));");
  __ Lbu(a1, MemOperand(a1));
  __ RecordComment("]");
  AdvanceBytecodeOffsetOrReturn(masm, kInterpreterBytecodeArrayRegister,
                                kInterpreterBytecodeOffsetRegister, a1, a2, a3,
                                a4, &do_return);
  __ RecordComment("[  Branch(&do_dispatch);");
  __ Branch(&do_dispatch);
  __ RecordComment("]");

  __ RecordComment("[  bind(&do_return);");
  __ bind(&do_return);
  __ RecordComment("]");
  // The return value is in a0.
  LeaveInterpreterFrame(masm, t0);
  __ RecordComment("[  Jump(ra);");
  __ Jump(ra);
  __ RecordComment("]");

  __ RecordComment("[  bind(&stack_check_interrupt);");
  __ bind(&stack_check_interrupt);
  __ RecordComment("]");
  // Modify the bytecode offset in the stack to be kFunctionEntryBytecodeOffset
  // for the call to the StackGuard.
  __ li(kInterpreterBytecodeOffsetRegister,
        Operand(Smi::FromInt(BytecodeArray::kHeaderSize - kHeapObjectTag +
                             kFunctionEntryBytecodeOffset)));
  __ Sd(kInterpreterBytecodeOffsetRegister,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeOffsetFromFp));
  __ RecordComment("[  CallRuntime(Runtime::kStackGuard);");
  __ CallRuntime(Runtime::kStackGuard);
  __ RecordComment("]");

  // After the call, restore the bytecode array, bytecode offset and accumulator
  // registers again. Also, restore the bytecode offset in the stack to its
  // previous value.
  __ Ld(kInterpreterBytecodeArrayRegister,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeArrayFromFp));
  __ li(kInterpreterBytecodeOffsetRegister,
        Operand(BytecodeArray::kHeaderSize - kHeapObjectTag));
  __ RecordComment(
      "[  LoadRoot(kInterpreterAccumulatorRegister, "
      "RootIndex::kUndefinedValue);");
  __ LoadRoot(kInterpreterAccumulatorRegister, RootIndex::kUndefinedValue);
  __ RecordComment("]");

  __ RecordComment("[  SmiTag(a5, kInterpreterBytecodeOffsetRegister);");
  __ SmiTag(a5, kInterpreterBytecodeOffsetRegister);
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(a5, MemOperand(fp, "
      "InterpreterFrameConstants::kBytecodeOffsetFromFp));");
  __ Sd(a5, MemOperand(fp, InterpreterFrameConstants::kBytecodeOffsetFromFp));
  __ RecordComment("]");

  __ RecordComment("[  Branch(&after_stack_check_interrupt);");
  __ Branch(&after_stack_check_interrupt);
  __ RecordComment("]");

  __ RecordComment("[  bind(&optimized_code_slot_not_empty);");
  __ bind(&optimized_code_slot_not_empty);
  __ RecordComment("]");
  Label maybe_has_optimized_code;
  // Check if optimized code marker is actually a weak reference to the
  // optimized code as opposed to an optimization marker.
  __ RecordComment(
      "[  JumpIfNotSmi(optimized_code_entry, &maybe_has_optimized_code);");
  __ JumpIfNotSmi(optimized_code_entry, &maybe_has_optimized_code);
  __ RecordComment("]");
  MaybeOptimizeCode(masm, feedback_vector, optimized_code_entry);
  // Fall through if there's no runnable optimized code.
  __ RecordComment("[  Branch(&not_optimized);");
  __ Branch(&not_optimized);
  __ RecordComment("]");

  __ RecordComment("[  bind(&maybe_has_optimized_code);");
  __ bind(&maybe_has_optimized_code);
  __ RecordComment("]");
  // Load code entry from the weak reference, if it was cleared, resume
  // execution of unoptimized code.
  __ RecordComment(
      "[  LoadWeakValue(optimized_code_entry, optimized_code_entry, "
      "&not_optimized);");
  __ LoadWeakValue(optimized_code_entry, optimized_code_entry, &not_optimized);
  __ RecordComment("]");
  TailCallOptimizedCodeSlot(masm, optimized_code_entry, t4, a5);

  __ RecordComment("[  bind(&compile_lazy);");
  __ bind(&compile_lazy);
  __ RecordComment("]");
  GenerateTailCallToReturnedCode(masm, Runtime::kCompileLazy);
  // Unreachable code.
  __ RecordComment("[  break_(0xCC);");
  __ break_(0xCC);
  __ RecordComment("]");

  __ RecordComment("[  bind(&stack_overflow);");
  __ bind(&stack_overflow);
  __ RecordComment("]");
  __ RecordComment("[  CallRuntime(Runtime::kThrowStackOverflow);");
  __ CallRuntime(Runtime::kThrowStackOverflow);
  __ RecordComment("]");
  // Unreachable code.
  __ RecordComment("[  break_(0xCC);");
  __ break_(0xCC);
  __ RecordComment("]");
}

static void Generate_InterpreterPushArgs(MacroAssembler* masm,
                                         Register num_args,
                                         Register start_address,
                                         Register scratch, Register scratch2) {
  // Find the address of the last argument.
  __ RecordComment("[  Sub64(scratch, num_args, Operand(1));");
  __ Sub64(scratch, num_args, Operand(1));
  __ RecordComment("]");
  __ RecordComment("[  Sll64(scratch, scratch, kSystemPointerSizeLog2);");
  __ Sll64(scratch, scratch, kSystemPointerSizeLog2);
  __ RecordComment("]");
  __ RecordComment("[  Sub64(start_address, start_address, scratch);");
  __ Sub64(start_address, start_address, scratch);
  __ RecordComment("]");

  // Push the arguments.
  __ PushArray(start_address, num_args, scratch, scratch2,
               TurboAssembler::PushArrayOrder::kReverse);
}

// static
void Builtins::Generate_InterpreterPushArgsThenCallImpl(
    MacroAssembler* masm, ConvertReceiverMode receiver_mode,
    InterpreterPushArgsMode mode) {
  DCHECK(mode != InterpreterPushArgsMode::kArrayFunction);
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a2 : the address of the first argument to be pushed. Subsequent
  //          arguments should be consecutive above this, in the same order as
  //          they are to be pushed onto the stack.
  //  -- a1 : the target to call (can be any Object).
  // -----------------------------------
  Label stack_overflow;
  if (mode == InterpreterPushArgsMode::kWithFinalSpread) {
    // The spread argument should not be pushed.
    __ RecordComment("[  Sub64(a0, a0, Operand(1));");
    __ Sub64(a0, a0, Operand(1));
    __ RecordComment("]");
  }

  __ RecordComment("[  Add64(a3, a0, Operand(1));  // Add one for receiver.");
  __ Add64(a3, a0, Operand(1));  // Add one for receiver.
  __ RecordComment("]");

  Generate_StackOverflowCheck(masm, a3, a4, t0, &stack_overflow);

  if (receiver_mode == ConvertReceiverMode::kNullOrUndefined) {
    // Don't copy receiver.
    __ RecordComment("[  Move(a3, a0);");
    __ Move(a3, a0);
    __ RecordComment("]");
  }

  // This function modifies a2, t0 and a4.
  Generate_InterpreterPushArgs(masm, a3, a2, a4, t0);

  if (receiver_mode == ConvertReceiverMode::kNullOrUndefined) {
    __ RecordComment("[  PushRoot(RootIndex::kUndefinedValue);");
    __ PushRoot(RootIndex::kUndefinedValue);
    __ RecordComment("]");
  }

  if (mode == InterpreterPushArgsMode::kWithFinalSpread) {
    // Pass the spread in the register a2.
    // a2 already points to the penultime argument, the spread
    // is below that.
    __ RecordComment("[  Ld(a2, MemOperand(a2, -kSystemPointerSize));");
    __ Ld(a2, MemOperand(a2, -kSystemPointerSize));
    __ RecordComment("]");
  }

  // Call the target.
  if (mode == InterpreterPushArgsMode::kWithFinalSpread) {
    __ Jump(BUILTIN_CODE(masm->isolate(), CallWithSpread),
            RelocInfo::CODE_TARGET);
  } else {
    __ Jump(masm->isolate()->builtins()->Call(ConvertReceiverMode::kAny),
            RelocInfo::CODE_TARGET);
  }

  __ RecordComment("[  bind(&stack_overflow);");
  __ bind(&stack_overflow);
  __ RecordComment("]");
  {
    __ RecordComment("[  TailCallRuntime(Runtime::kThrowStackOverflow);");
    __ TailCallRuntime(Runtime::kThrowStackOverflow);
    __ RecordComment("]");
    // Unreachable code.
    __ RecordComment("[  break_(0xCC);");
    __ break_(0xCC);
    __ RecordComment("]");
  }
}

// static
void Builtins::Generate_InterpreterPushArgsThenConstructImpl(
    MacroAssembler* masm, InterpreterPushArgsMode mode) {
  // ----------- S t a t e -------------
  // -- a0 : argument count (not including receiver)
  // -- a3 : new target
  // -- a1 : constructor to call
  // -- a2 : allocation site feedback if available, undefined otherwise.
  // -- a4 : address of the first argument
  // -----------------------------------
  Label stack_overflow;
  __ RecordComment("[  Add64(a6, a0, 1);");
  __ Add64(a6, a0, 1);
  __ RecordComment("]");
  Generate_StackOverflowCheck(masm, a6, a5, t0, &stack_overflow);

  if (mode == InterpreterPushArgsMode::kWithFinalSpread) {
    // The spread argument should not be pushed.
    __ RecordComment("[  Sub64(a0, a0, Operand(1));");
    __ Sub64(a0, a0, Operand(1));
    __ RecordComment("]");
  }

  // Push the arguments, This function modifies t0, a4 and a5.
  Generate_InterpreterPushArgs(masm, a0, a4, a5, t0);

  // Push a slot for the receiver.
  __ RecordComment("[  push(zero_reg);");
  __ push(zero_reg);
  __ RecordComment("]");

  if (mode == InterpreterPushArgsMode::kWithFinalSpread) {
    // Pass the spread in the register a2.
    // a4 already points to the penultimate argument, the spread
    // lies in the next interpreter register.
    __ RecordComment("[  Ld(a2, MemOperand(a4, -kSystemPointerSize));");
    __ Ld(a2, MemOperand(a4, -kSystemPointerSize));
    __ RecordComment("]");
  } else {
    __ RecordComment("[  AssertUndefinedOrAllocationSite(a2, t0);");
    __ AssertUndefinedOrAllocationSite(a2, t0);
    __ RecordComment("]");
  }

  if (mode == InterpreterPushArgsMode::kArrayFunction) {
    __ RecordComment("[  AssertFunction(a1);");
    __ AssertFunction(a1);
    __ RecordComment("]");

    // Tail call to the function-specific construct stub (still in the caller
    // context at this point).
    __ Jump(BUILTIN_CODE(masm->isolate(), ArrayConstructorImpl),
            RelocInfo::CODE_TARGET);
  } else if (mode == InterpreterPushArgsMode::kWithFinalSpread) {
    // Call the constructor with a0, a1, and a3 unmodified.
    __ Jump(BUILTIN_CODE(masm->isolate(), ConstructWithSpread),
            RelocInfo::CODE_TARGET);
  } else {
    DCHECK_EQ(InterpreterPushArgsMode::kOther, mode);
    // Call the constructor with a0, a1, and a3 unmodified.
    __ RecordComment(
        "[  Jump(BUILTIN_CODE(masm->isolate(), Construct), "
        "RelocInfo::CODE_TARGET);");
    __ Jump(BUILTIN_CODE(masm->isolate(), Construct), RelocInfo::CODE_TARGET);
    __ RecordComment("]");
  }

  __ RecordComment("[  bind(&stack_overflow);");
  __ bind(&stack_overflow);
  __ RecordComment("]");
  {
    __ RecordComment("[  TailCallRuntime(Runtime::kThrowStackOverflow);");
    __ TailCallRuntime(Runtime::kThrowStackOverflow);
    __ RecordComment("]");
    // Unreachable code.
    __ RecordComment("[  break_(0xCC);");
    __ break_(0xCC);
    __ RecordComment("]");
  }
}

static void Generate_InterpreterEnterBytecode(MacroAssembler* masm) {
  // Set the return address to the correct point in the interpreter entry
  // trampoline.
  Label builtin_trampoline, trampoline_loaded;
  Smi interpreter_entry_return_pc_offset(
      masm->isolate()->heap()->interpreter_entry_return_pc_offset());
  DCHECK_NE(interpreter_entry_return_pc_offset, Smi::zero());

  // If the SFI function_data is an InterpreterData, the function will have a
  // custom copy of the interpreter entry trampoline for profiling. If so,
  // get the custom trampoline, otherwise grab the entry address of the global
  // trampoline.
  __ RecordComment(
      "[  Ld(t0, MemOperand(fp, StandardFrameConstants::kFunctionOffset));");
  __ Ld(t0, MemOperand(fp, StandardFrameConstants::kFunctionOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  LoadTaggedPointerField(t0, FieldMemOperand(t0, "
      "JSFunction::kSharedFunctionInfoOffset));");
  __ LoadTaggedPointerField(
      t0, FieldMemOperand(t0, JSFunction::kSharedFunctionInfoOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  LoadTaggedPointerField(t0, FieldMemOperand(t0, "
      "SharedFunctionInfo::kFunctionDataOffset));");
  __ LoadTaggedPointerField(
      t0, FieldMemOperand(t0, SharedFunctionInfo::kFunctionDataOffset));
  __ RecordComment("]");
  __ GetObjectType(t0, kInterpreterDispatchTableRegister,
                   kInterpreterDispatchTableRegister);
  __ Branch(&builtin_trampoline, ne, kInterpreterDispatchTableRegister,
            Operand(INTERPRETER_DATA_TYPE));

  __ RecordComment(
      "[  LoadTaggedPointerField(t0, FieldMemOperand(t0, "
      "InterpreterData::kInterpreterTrampolineOffset));");
  __ LoadTaggedPointerField(
      t0, FieldMemOperand(t0, InterpreterData::kInterpreterTrampolineOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Add64(t0, t0, Operand(Code::kHeaderSize - kHeapObjectTag));");
  __ Add64(t0, t0, Operand(Code::kHeaderSize - kHeapObjectTag));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&trampoline_loaded);");
  __ Branch(&trampoline_loaded);
  __ RecordComment("]");

  __ RecordComment("[  bind(&builtin_trampoline);");
  __ bind(&builtin_trampoline);
  __ RecordComment("]");
  __ li(t0, ExternalReference::
                address_of_interpreter_entry_trampoline_instruction_start(
                    masm->isolate()));
  __ RecordComment("[  Ld(t0, MemOperand(t0));");
  __ Ld(t0, MemOperand(t0));
  __ RecordComment("]");

  __ RecordComment("[  bind(&trampoline_loaded);");
  __ bind(&trampoline_loaded);
  __ RecordComment("]");
  __ RecordComment(
      "[  Add64(ra, t0, Operand(interpreter_entry_return_pc_offset.value()));");
  __ Add64(ra, t0, Operand(interpreter_entry_return_pc_offset.value()));
  __ RecordComment("]");

  // Initialize the dispatch table register.
  __ li(kInterpreterDispatchTableRegister,
        ExternalReference::interpreter_dispatch_table_address(masm->isolate()));

  // Get the bytecode array pointer from the frame.
  __ Ld(kInterpreterBytecodeArrayRegister,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeArrayFromFp));

  if (FLAG_debug_code) {
    // Check function data field is actually a BytecodeArray object.
    __ RecordComment(
        "[  SmiTst(kInterpreterBytecodeArrayRegister, kScratchReg);");
    __ SmiTst(kInterpreterBytecodeArrayRegister, kScratchReg);
    __ RecordComment("]");
    __ Assert(ne,
              AbortReason::kFunctionDataShouldBeBytecodeArrayOnInterpreterEntry,
              kScratchReg, Operand(zero_reg));
    __ RecordComment(
        "[  GetObjectType(kInterpreterBytecodeArrayRegister, a1, a1);");
    __ GetObjectType(kInterpreterBytecodeArrayRegister, a1, a1);
    __ RecordComment("]");
    __ Assert(eq,
              AbortReason::kFunctionDataShouldBeBytecodeArrayOnInterpreterEntry,
              a1, Operand(BYTECODE_ARRAY_TYPE));
  }

  // Get the target bytecode offset from the frame.
  __ SmiUntag(kInterpreterBytecodeOffsetRegister,
              MemOperand(fp, InterpreterFrameConstants::kBytecodeOffsetFromFp));

  if (FLAG_debug_code) {
    Label okay;
    __ Branch(&okay, ge, kInterpreterBytecodeOffsetRegister,
              Operand(BytecodeArray::kHeaderSize - kHeapObjectTag));
    // Unreachable code.
    __ RecordComment("[  break_(0xCC);");
    __ break_(0xCC);
    __ RecordComment("]");
    __ RecordComment("[  bind(&okay);");
    __ bind(&okay);
    __ RecordComment("]");
  }

  // Dispatch to the target bytecode.
  __ Add64(a1, kInterpreterBytecodeArrayRegister,
           kInterpreterBytecodeOffsetRegister);
  __ RecordComment("[  Lbu(a7, MemOperand(a1));");
  __ Lbu(a7, MemOperand(a1));
  __ RecordComment("]");
  __ CalcScaledAddress(a1, kInterpreterDispatchTableRegister, a7,
                       kSystemPointerSizeLog2);
  __ RecordComment("[  Ld(kJavaScriptCallCodeStartRegister, MemOperand(a1));");
  __ Ld(kJavaScriptCallCodeStartRegister, MemOperand(a1));
  __ RecordComment("]");
  __ RecordComment("[  Jump(kJavaScriptCallCodeStartRegister);");
  __ Jump(kJavaScriptCallCodeStartRegister);
  __ RecordComment("]");
}

void Builtins::Generate_InterpreterEnterBytecodeAdvance(MacroAssembler* masm) {
  // Advance the current bytecode offset stored within the given interpreter
  // stack frame. This simulates what all bytecode handlers do upon completion
  // of the underlying operation.
  __ Ld(kInterpreterBytecodeArrayRegister,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeArrayFromFp));
  __ Ld(kInterpreterBytecodeOffsetRegister,
        MemOperand(fp, InterpreterFrameConstants::kBytecodeOffsetFromFp));
  __ RecordComment("[  SmiUntag(kInterpreterBytecodeOffsetRegister);");
  __ SmiUntag(kInterpreterBytecodeOffsetRegister);
  __ RecordComment("]");

  Label enter_bytecode, function_entry_bytecode;
  __ Branch(&function_entry_bytecode, eq, kInterpreterBytecodeOffsetRegister,
            Operand(BytecodeArray::kHeaderSize - kHeapObjectTag +
                    kFunctionEntryBytecodeOffset));

  // Load the current bytecode.
  __ Add64(a1, kInterpreterBytecodeArrayRegister,
           kInterpreterBytecodeOffsetRegister);
  __ RecordComment("[  Lbu(a1, MemOperand(a1));");
  __ Lbu(a1, MemOperand(a1));
  __ RecordComment("]");

  // Advance to the next bytecode.
  Label if_return;
  AdvanceBytecodeOffsetOrReturn(masm, kInterpreterBytecodeArrayRegister,
                                kInterpreterBytecodeOffsetRegister, a1, a2, a3,
                                a4, &if_return);

  __ RecordComment("[  bind(&enter_bytecode);");
  __ bind(&enter_bytecode);
  __ RecordComment("]");
  // Convert new bytecode offset to a Smi and save in the stackframe.
  __ RecordComment("[  SmiTag(a2, kInterpreterBytecodeOffsetRegister);");
  __ SmiTag(a2, kInterpreterBytecodeOffsetRegister);
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(a2, MemOperand(fp, "
      "InterpreterFrameConstants::kBytecodeOffsetFromFp));");
  __ Sd(a2, MemOperand(fp, InterpreterFrameConstants::kBytecodeOffsetFromFp));
  __ RecordComment("]");

  Generate_InterpreterEnterBytecode(masm);

  __ RecordComment("[  bind(&function_entry_bytecode);");
  __ bind(&function_entry_bytecode);
  __ RecordComment("]");
  // If the code deoptimizes during the implicit function entry stack interrupt
  // check, it will have a bailout ID of kFunctionEntryBytecodeOffset, which is
  // not a valid bytecode offset. Detect this case and advance to the first
  // actual bytecode.
  __ li(kInterpreterBytecodeOffsetRegister,
        Operand(BytecodeArray::kHeaderSize - kHeapObjectTag));
  __ RecordComment("[  Branch(&enter_bytecode);");
  __ Branch(&enter_bytecode);
  __ RecordComment("]");

  // We should never take the if_return path.
  __ RecordComment("[  bind(&if_return);");
  __ bind(&if_return);
  __ RecordComment("]");
  __ RecordComment("[  Abort(AbortReason::kInvalidBytecodeAdvance);");
  __ Abort(AbortReason::kInvalidBytecodeAdvance);
  __ RecordComment("]");
}

void Builtins::Generate_InterpreterEnterBytecodeDispatch(MacroAssembler* masm) {
  Generate_InterpreterEnterBytecode(masm);
}

namespace {
void Generate_ContinueToBuiltinHelper(MacroAssembler* masm,
                                      bool java_script_builtin,
                                      bool with_result) {
  const RegisterConfiguration* config(RegisterConfiguration::Default());
  int allocatable_register_count = config->num_allocatable_general_registers();
  Register scratch = t3;
  if (with_result) {
    if (java_script_builtin) {
      __ RecordComment("[  Move(scratch, a0);");
      __ Move(scratch, a0);
      __ RecordComment("]");
    } else {
      // Overwrite the hole inserted by the deoptimizer with the return value
      // from the LAZY deopt point.
      __ Sd(a0,
            MemOperand(sp,
                       config->num_allocatable_general_registers() *
                               kSystemPointerSize +
                           BuiltinContinuationFrameConstants::kFixedFrameSize));
    }
  }
  for (int i = allocatable_register_count - 1; i >= 0; --i) {
    int code = config->GetAllocatableGeneralCode(i);
    __ RecordComment("[  Pop(Register::from_code(code));");
    __ Pop(Register::from_code(code));
    __ RecordComment("]");
    if (java_script_builtin && code == kJavaScriptCallArgCountRegister.code()) {
      __ RecordComment("[  SmiUntag(Register::from_code(code));");
      __ SmiUntag(Register::from_code(code));
      __ RecordComment("]");
    }
  }

  if (with_result && java_script_builtin) {
    // Overwrite the hole inserted by the deoptimizer with the return value from
    // the LAZY deopt point. t0 contains the arguments count, the return value
    // from LAZY is always the last argument.
    __ Add64(a0, a0,
             Operand(BuiltinContinuationFrameConstants::kFixedSlotCount));
    __ RecordComment(
        "[  CalcScaledAddress(t0, sp, a0, kSystemPointerSizeLog2);");
    __ CalcScaledAddress(t0, sp, a0, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sd(scratch, MemOperand(t0));");
    __ Sd(scratch, MemOperand(t0));
    __ RecordComment("]");
    // Recover arguments count.
    __ Sub64(a0, a0,
             Operand(BuiltinContinuationFrameConstants::kFixedSlotCount));
  }

  __ Ld(fp, MemOperand(
                sp, BuiltinContinuationFrameConstants::kFixedFrameSizeFromFp));
  // Load builtin index (stored as a Smi) and use it to get the builtin start
  // address from the builtins table.
  __ RecordComment("[  Pop(t0);");
  __ Pop(t0);
  __ RecordComment("]");
  __ Add64(sp, sp,
           Operand(BuiltinContinuationFrameConstants::kFixedFrameSizeFromFp));
  __ RecordComment("[  Pop(ra);");
  __ Pop(ra);
  __ RecordComment("]");
  __ RecordComment("[  LoadEntryFromBuiltinIndex(t0);");
  __ LoadEntryFromBuiltinIndex(t0);
  __ RecordComment("]");
  __ RecordComment("[  Jump(t0);");
  __ Jump(t0);
  __ RecordComment("]");
}
}  // namespace

void Builtins::Generate_ContinueToCodeStubBuiltin(MacroAssembler* masm) {
  Generate_ContinueToBuiltinHelper(masm, false, false);
}

void Builtins::Generate_ContinueToCodeStubBuiltinWithResult(
    MacroAssembler* masm) {
  Generate_ContinueToBuiltinHelper(masm, false, true);
}

void Builtins::Generate_ContinueToJavaScriptBuiltin(MacroAssembler* masm) {
  Generate_ContinueToBuiltinHelper(masm, true, false);
}

void Builtins::Generate_ContinueToJavaScriptBuiltinWithResult(
    MacroAssembler* masm) {
  Generate_ContinueToBuiltinHelper(masm, true, true);
}

void Builtins::Generate_NotifyDeoptimized(MacroAssembler* masm) {
  {
    FrameScope scope(masm, StackFrame::INTERNAL);
    __ RecordComment("[  CallRuntime(Runtime::kNotifyDeoptimized);");
    __ CallRuntime(Runtime::kNotifyDeoptimized);
    __ RecordComment("]");
  }

  DCHECK_EQ(kInterpreterAccumulatorRegister.code(), a0.code());
  __ RecordComment("[  Ld(a0, MemOperand(sp, 0 * kSystemPointerSize));");
  __ Ld(a0, MemOperand(sp, 0 * kSystemPointerSize));
  __ RecordComment("]");
  __ RecordComment(
      "[  Add64(sp, sp, Operand(1 * kSystemPointerSize));  // Remove state.");
  __ Add64(sp, sp, Operand(1 * kSystemPointerSize));  // Remove state.
  __ RecordComment("]");
  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");
}

void Builtins::Generate_InterpreterOnStackReplacement(MacroAssembler* masm) {
  {
    FrameScope scope(masm, StackFrame::INTERNAL);
    __ RecordComment("[  CallRuntime(Runtime::kCompileForOnStackReplacement);");
    __ CallRuntime(Runtime::kCompileForOnStackReplacement);
    __ RecordComment("]");
  }

  // If the code object is null, just return to the caller.
  __ RecordComment("[  Ret(eq, a0, Operand(Smi::zero()));");
  __ Ret(eq, a0, Operand(Smi::zero()));
  __ RecordComment("]");

  // Drop the handler frame that is be sitting on top of the actual
  // JavaScript frame. This is the case then OSR is triggered from bytecode.
  __ RecordComment("[  LeaveFrame(StackFrame::STUB);");
  __ LeaveFrame(StackFrame::STUB);
  __ RecordComment("]");

  // Load deoptimization data from the code object.
  // <deopt_data> = <code>[#deoptimization_data_offset]
  __ RecordComment(
      "[  LoadTaggedPointerField(a1, MemOperand(a0, "
      "Code::kDeoptimizationDataOffset - kHeapObjectTag));");
  __ LoadTaggedPointerField(
      a1, MemOperand(a0, Code::kDeoptimizationDataOffset - kHeapObjectTag));
  __ RecordComment("]");

  // Load the OSR entrypoint offset from the deoptimization data.
  // <osr_offset> = <deopt_data>[#header_size + #osr_pc_offset]
  __ SmiUntag(a1, MemOperand(a1, FixedArray::OffsetOfElementAt(
                                     DeoptimizationData::kOsrPcOffsetIndex) -
                                     kHeapObjectTag));

  // Compute the target address = code_obj + header_size + osr_offset
  // <entry_addr> = <code_obj> + #header_size + <osr_offset>
  __ RecordComment("[  Add64(a0, a0, a1);");
  __ Add64(a0, a0, a1);
  __ RecordComment("]");
  __ RecordComment("[  Add64(ra, a0, Code::kHeaderSize - kHeapObjectTag);");
  __ Add64(ra, a0, Code::kHeaderSize - kHeapObjectTag);
  __ RecordComment("]");
  // And "return" to the OSR entry point of the function.
  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");
}

// static
void Builtins::Generate_FunctionPrototypeApply(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0    : argc
  //  -- sp[0] : receiver
  //  -- sp[4] : thisArg
  //  -- sp[8] : argArray
  // -----------------------------------

  Register argc = a0;
  Register arg_array = a2;
  Register receiver = a1;
  Register this_arg = a5;
  Register undefined_value = a3;

  __ RecordComment("[  LoadRoot(undefined_value, RootIndex::kUndefinedValue);");
  __ LoadRoot(undefined_value, RootIndex::kUndefinedValue);
  __ RecordComment("]");

  // 1. Load receiver into a1, argArray into a2 (if present), remove all
  // arguments from the stack (including the receiver), and push thisArg (if
  // present) instead.
  {
    // Claim (2 - argc) dummy arguments form the stack, to put the stack in a
    // consistent state for a simple pop operation.

    __ RecordComment("[  Ld(this_arg, MemOperand(sp, kSystemPointerSize));");
    __ Ld(this_arg, MemOperand(sp, kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(arg_array, MemOperand(sp, 2 * kSystemPointerSize));");
    __ Ld(arg_array, MemOperand(sp, 2 * kSystemPointerSize));
    __ RecordComment("]");

    Label done0, done1;
    __ RecordComment("[  Branch(&done0, ne, argc, Operand(zero_reg));");
    __ Branch(&done0, ne, argc, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment("[  Move(arg_array, undefined_value);  // if argc == 0");
    __ Move(arg_array, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment("[  Move(this_arg, undefined_value);   // if argc == 0");
    __ Move(this_arg, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment(
        "[  bind(&done0);                      // else (i.e., argc > 0)");
    __ bind(&done0);  // else (i.e., argc > 0)
    __ RecordComment("]");

    __ RecordComment("[  Branch(&done1, ne, argc, Operand(1));");
    __ Branch(&done1, ne, argc, Operand(1));
    __ RecordComment("]");
    __ RecordComment("[  Move(arg_array, undefined_value);  // if argc == 1");
    __ Move(arg_array, undefined_value);  // if argc == 1
    __ RecordComment("]");
    __ RecordComment(
        "[  bind(&done1);                      // else (i.e., argc > 1)");
    __ bind(&done1);  // else (i.e., argc > 1)
    __ RecordComment("]");

    __ RecordComment("[  Ld(receiver, MemOperand(sp));");
    __ Ld(receiver, MemOperand(sp));
    __ RecordComment("]");
    __ RecordComment("[  Sd(this_arg, MemOperand(sp));");
    __ Sd(this_arg, MemOperand(sp));
    __ RecordComment("]");
  }

  // ----------- S t a t e -------------
  //  -- a2    : argArray
  //  -- a1    : receiver
  //  -- a3    : undefined root value
  //  -- sp[0] : thisArg
  // -----------------------------------

  // 2. We don't need to check explicitly for callable receiver here,
  // since that's the first thing the Call/CallWithArrayLike builtins
  // will do.

  // 3. Tail call with no arguments if argArray is null or undefined.
  Label no_arguments;
  __ RecordComment(
      "[  JumpIfRoot(arg_array, RootIndex::kNullValue, &no_arguments);");
  __ JumpIfRoot(arg_array, RootIndex::kNullValue, &no_arguments);
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&no_arguments, eq, arg_array, Operand(undefined_value));");
  __ Branch(&no_arguments, eq, arg_array, Operand(undefined_value));
  __ RecordComment("]");

  // 4a. Apply the receiver to the given argArray.
  __ Jump(BUILTIN_CODE(masm->isolate(), CallWithArrayLike),
          RelocInfo::CODE_TARGET);

  // 4b. The argArray is either null or undefined, so we tail call without any
  // arguments to the receiver.
  __ RecordComment("[  bind(&no_arguments);");
  __ bind(&no_arguments);
  __ RecordComment("]");
  {
    __ RecordComment("[  Move(a0, zero_reg);");
    __ Move(a0, zero_reg);
    __ RecordComment("]");
    DCHECK(receiver == a1);
    __ RecordComment(
        "[  Jump(masm->isolate()->builtins()->Call(), "
        "RelocInfo::CODE_TARGET);");
    __ Jump(masm->isolate()->builtins()->Call(), RelocInfo::CODE_TARGET);
    __ RecordComment("]");
  }
}

// static
void Builtins::Generate_FunctionPrototypeCall(MacroAssembler* masm) {
  // 1. Get the callable to call (passed as receiver) from the stack.
  __ RecordComment("[  Pop(a1); }");
  { __ Pop(a1); }
  __ RecordComment("]");

  // 2. Make sure we have at least one argument.
  // a0: actual number of arguments
  {
    Label done;
    __ RecordComment("[  Branch(&done, ne, a0, Operand(zero_reg));");
    __ Branch(&done, ne, a0, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment("[  PushRoot(RootIndex::kUndefinedValue);");
    __ PushRoot(RootIndex::kUndefinedValue);
    __ RecordComment("]");
    __ RecordComment("[  Add64(a0, a0, Operand(1));");
    __ Add64(a0, a0, Operand(1));
    __ RecordComment("]");
    __ RecordComment("[  bind(&done);");
    __ bind(&done);
    __ RecordComment("]");
  }

  // 3. Adjust the actual number of arguments.
  __ RecordComment("[  Add64(a0, a0, -1);");
  __ Add64(a0, a0, -1);
  __ RecordComment("]");

  // 4. Call the callable.
  __ RecordComment(
      "[  Jump(masm->isolate()->builtins()->Call(), RelocInfo::CODE_TARGET);");
  __ Jump(masm->isolate()->builtins()->Call(), RelocInfo::CODE_TARGET);
  __ RecordComment("]");
}

void Builtins::Generate_ReflectApply(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0     : argc
  //  -- sp[0]  : receiver
  //  -- sp[8]  : target         (if argc >= 1)
  //  -- sp[16] : thisArgument   (if argc >= 2)
  //  -- sp[24] : argumentsList  (if argc == 3)
  // -----------------------------------

  Register argc = a0;
  Register arguments_list = a2;
  Register target = a1;
  Register this_argument = a5;
  Register undefined_value = a3;

  __ RecordComment("[  LoadRoot(undefined_value, RootIndex::kUndefinedValue);");
  __ LoadRoot(undefined_value, RootIndex::kUndefinedValue);
  __ RecordComment("]");

  // 1. Load target into a1 (if present), argumentsList into a2 (if present),
  // remove all arguments from the stack (including the receiver), and push
  // thisArgument (if present) instead.
  {
    // Claim (3 - argc) dummy arguments form the stack, to put the stack in a
    // consistent state for a simple pop operation.

    __ RecordComment("[  Ld(target, MemOperand(sp, kSystemPointerSize));");
    __ Ld(target, MemOperand(sp, kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(this_argument, MemOperand(sp, 2 * kSystemPointerSize));");
    __ Ld(this_argument, MemOperand(sp, 2 * kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(arguments_list, MemOperand(sp, 3 * kSystemPointerSize));");
    __ Ld(arguments_list, MemOperand(sp, 3 * kSystemPointerSize));
    __ RecordComment("]");

    Label done0, done1, done2;
    __ RecordComment("[  Branch(&done0, ne, argc, Operand(zero_reg));");
    __ Branch(&done0, ne, argc, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(arguments_list, undefined_value);  // if argc == 0");
    __ Move(arguments_list, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(this_argument, undefined_value);   // if argc == 0");
    __ Move(this_argument, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(target, undefined_value);          // if argc == 0");
    __ Move(target, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment("[  bind(&done0);                           // argc != 0");
    __ bind(&done0);  // argc != 0
    __ RecordComment("]");

    __ RecordComment("[  Branch(&done1, ne, argc, Operand(1));");
    __ Branch(&done1, ne, argc, Operand(1));
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(arguments_list, undefined_value);  // if argc == 1");
    __ Move(arguments_list, undefined_value);  // if argc == 1
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(this_argument, undefined_value);   // if argc == 1");
    __ Move(this_argument, undefined_value);  // if argc == 1
    __ RecordComment("]");
    __ RecordComment("[  bind(&done1);                           // argc > 1");
    __ bind(&done1);  // argc > 1
    __ RecordComment("]");

    __ RecordComment("[  Branch(&done2, ne, argc, Operand(2));");
    __ Branch(&done2, ne, argc, Operand(2));
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(arguments_list, undefined_value);  // if argc == 2");
    __ Move(arguments_list, undefined_value);  // if argc == 2
    __ RecordComment("]");
    __ RecordComment("[  bind(&done2);                           // argc > 2");
    __ bind(&done2);  // argc > 2
    __ RecordComment("]");

    __ RecordComment(
        "[  Sd(this_argument, MemOperand(sp, 0));  // Overwrite receiver");
    __ Sd(this_argument, MemOperand(sp, 0));  // Overwrite receiver
    __ RecordComment("]");
  }

  // ----------- S t a t e -------------
  //  -- a2    : argumentsList
  //  -- a1    : target
  //  -- a3    : undefined root value
  //  -- sp[0] : thisArgument
  // -----------------------------------

  // 2. We don't need to check explicitly for callable target here,
  // since that's the first thing the Call/CallWithArrayLike builtins
  // will do.

  // 3. Apply the target to the given argumentsList.
  __ Jump(BUILTIN_CODE(masm->isolate(), CallWithArrayLike),
          RelocInfo::CODE_TARGET);
}

void Builtins::Generate_ReflectConstruct(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0     : argc
  //  -- sp[0]   : receiver
  //  -- sp[8]   : target
  //  -- sp[16]  : argumentsList
  //  -- sp[24]  : new.target (optional)
  // -----------------------------------
  Register argc = a0;
  Register arguments_list = a2;
  Register target = a1;
  Register new_target = a3;
  Register undefined_value = a4;

  __ RecordComment("[  LoadRoot(undefined_value, RootIndex::kUndefinedValue);");
  __ LoadRoot(undefined_value, RootIndex::kUndefinedValue);
  __ RecordComment("]");

  // 1. Load target into a1 (if present), argumentsList into a2 (if present),
  // new.target into a3 (if present, otherwise use target), remove all
  // arguments from the stack (including the receiver), and push thisArgument
  // (if present) instead.
  {
    // Claim (3 - argc) dummy arguments form the stack, to put the stack in a
    // consistent state for a simple pop operation.
    __ RecordComment("[  Ld(target, MemOperand(sp, kSystemPointerSize));");
    __ Ld(target, MemOperand(sp, kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(arguments_list, MemOperand(sp, 2 * kSystemPointerSize));");
    __ Ld(arguments_list, MemOperand(sp, 2 * kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(new_target, MemOperand(sp, 3 * kSystemPointerSize));");
    __ Ld(new_target, MemOperand(sp, 3 * kSystemPointerSize));
    __ RecordComment("]");

    Label done0, done1, done2;
    __ RecordComment("[  Branch(&done0, ne, argc, Operand(zero_reg));");
    __ Branch(&done0, ne, argc, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(arguments_list, undefined_value);  // if argc == 0");
    __ Move(arguments_list, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(new_target, undefined_value);      // if argc == 0");
    __ Move(new_target, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(target, undefined_value);          // if argc == 0");
    __ Move(target, undefined_value);  // if argc == 0
    __ RecordComment("]");
    __ RecordComment("[  bind(&done0);");
    __ bind(&done0);
    __ RecordComment("]");

    __ RecordComment("[  Branch(&done1, ne, argc, Operand(1));");
    __ Branch(&done1, ne, argc, Operand(1));
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(arguments_list, undefined_value);  // if argc == 1");
    __ Move(arguments_list, undefined_value);  // if argc == 1
    __ RecordComment("]");
    __ RecordComment(
        "[  Move(new_target, target);               // if argc == 1");
    __ Move(new_target, target);  // if argc == 1
    __ RecordComment("]");
    __ RecordComment("[  bind(&done1);");
    __ bind(&done1);
    __ RecordComment("]");

    __ RecordComment("[  Branch(&done2, ne, argc, Operand(2));");
    __ Branch(&done2, ne, argc, Operand(2));
    __ RecordComment("]");
    __ RecordComment("[  Move(new_target, target);  // if argc == 2");
    __ Move(new_target, target);  // if argc == 2
    __ RecordComment("]");
    __ RecordComment("[  bind(&done2);");
    __ bind(&done2);
    __ RecordComment("]");

    __ RecordComment(
        "[  Sd(undefined_value, MemOperand(sp, 0));  // Overwrite receiver");
    __ Sd(undefined_value, MemOperand(sp, 0));  // Overwrite receiver
    __ RecordComment("]");
  }

  // ----------- S t a t e -------------
  //  -- a2    : argumentsList
  //  -- a1    : target
  //  -- a3    : new.target
  //  -- sp[0] : receiver (undefined)
  // -----------------------------------

  // 2. We don't need to check explicitly for constructor target here,
  // since that's the first thing the Construct/ConstructWithArrayLike
  // builtins will do.

  // 3. We don't need to check explicitly for constructor new.target here,
  // since that's the second thing the Construct/ConstructWithArrayLike
  // builtins will do.

  // 4. Construct the target with the given new.target and argumentsList.
  __ Jump(BUILTIN_CODE(masm->isolate(), ConstructWithArrayLike),
          RelocInfo::CODE_TARGET);
}

static void EnterArgumentsAdaptorFrame(MacroAssembler* masm) {
  __ RecordComment("[  SmiTag(a0);");
  __ SmiTag(a0);
  __ RecordComment("]");
  __ RecordComment(
      "[  li(a4, "
      "Operand(StackFrame::TypeToMarker(StackFrame::ARGUMENTS_ADAPTOR)));");
  __ li(a4, Operand(StackFrame::TypeToMarker(StackFrame::ARGUMENTS_ADAPTOR)));
  __ RecordComment("]");
  __ RecordComment(
      "[  MultiPush(a0.bit() | a1.bit() | a4.bit() | fp.bit() | ra.bit());");
  __ MultiPush(a0.bit() | a1.bit() | a4.bit() | fp.bit() | ra.bit());
  __ RecordComment("]");
  __ RecordComment("[  Push(Smi::zero());  // Padding.");
  __ Push(Smi::zero());  // Padding.
  __ RecordComment("]");
  __ Add64(fp, sp,
           Operand(ArgumentsAdaptorFrameConstants::kFixedFrameSizeFromFp));
}

static void LeaveArgumentsAdaptorFrame(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0 : result being passed through
  // -----------------------------------
  // Get the number of arguments passed (as a smi), tear down the frame and
  // then tear down the parameters.
  __ RecordComment(
      "[  Ld(a1, MemOperand(fp, "
      "ArgumentsAdaptorFrameConstants::kLengthOffset));");
  __ Ld(a1, MemOperand(fp, ArgumentsAdaptorFrameConstants::kLengthOffset));
  __ RecordComment("]");
  __ RecordComment("[  Move(sp, fp);");
  __ Move(sp, fp);
  __ RecordComment("]");
  __ RecordComment("[  MultiPop(fp.bit() | ra.bit());");
  __ MultiPop(fp.bit() | ra.bit());
  __ RecordComment("]");
  __ RecordComment("[  SmiScale(a4, a1, kSystemPointerSizeLog2);");
  __ SmiScale(a4, a1, kSystemPointerSizeLog2);
  __ RecordComment("]");
  __ RecordComment("[  Add64(sp, sp, a4);");
  __ Add64(sp, sp, a4);
  __ RecordComment("]");
  // Adjust for the receiver.
  __ RecordComment("[  Add64(sp, sp, Operand(kSystemPointerSize));");
  __ Add64(sp, sp, Operand(kSystemPointerSize));
  __ RecordComment("]");
}

// static
void Builtins::Generate_CallOrConstructVarargs(MacroAssembler* masm,
                                               Handle<Code> code) {
  // ----------- S t a t e -------------
  //  -- a1 : target
  //  -- a0 : number of parameters on the stack (not including the receiver)
  //  -- a2 : arguments list (a FixedArray)
  //  -- a4 : len (number of elements to push from args)
  //  -- a3 : new.target (for [[Construct]])
  // -----------------------------------
  if (masm->emit_debug_code()) {
    // Allow a2 to be a FixedArray, or a FixedDoubleArray if a4 == 0.
    Label ok, fail;
    __ RecordComment("[  AssertNotSmi(a2);");
    __ AssertNotSmi(a2);
    __ RecordComment("]");
    __ RecordComment("[  GetObjectType(a2, t5, t5);");
    __ GetObjectType(a2, t5, t5);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&ok, eq, t5, Operand(FIXED_ARRAY_TYPE));");
    __ Branch(&ok, eq, t5, Operand(FIXED_ARRAY_TYPE));
    __ RecordComment("]");
    __ RecordComment(
        "[  Branch(&fail, ne, t5, Operand(FIXED_DOUBLE_ARRAY_TYPE));");
    __ Branch(&fail, ne, t5, Operand(FIXED_DOUBLE_ARRAY_TYPE));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&ok, eq, a4, Operand(zero_reg));");
    __ Branch(&ok, eq, a4, Operand(zero_reg));
    __ RecordComment("]");
    // Fall through.
    __ RecordComment("[  bind(&fail);");
    __ bind(&fail);
    __ RecordComment("]");
    __ RecordComment("[  Abort(AbortReason::kOperandIsNotAFixedArray);");
    __ Abort(AbortReason::kOperandIsNotAFixedArray);
    __ RecordComment("]");

    __ RecordComment("[  bind(&ok);");
    __ bind(&ok);
    __ RecordComment("]");
  }

  Register args = a2;
  Register len = a4;

  // Check for stack overflow.
  Label stack_overflow;
  Generate_StackOverflowCheck(masm, len, kScratchReg, a5, &stack_overflow);

  // Move the arguments already in the stack,
  // including the receiver and the return address.
  {
    Label copy;
    Register src = a6, dest = a7;
    __ RecordComment("[  Move(src, sp);");
    __ Move(src, sp);
    __ RecordComment("]");
    __ RecordComment("[  Sll64(t0, a4, kSystemPointerSizeLog2);");
    __ Sll64(t0, a4, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(sp, sp, Operand(t0));");
    __ Sub64(sp, sp, Operand(t0));
    __ RecordComment("]");
    // Update stack pointer.
    __ RecordComment("[  Move(dest, sp);");
    __ Move(dest, sp);
    __ RecordComment("]");
    __ RecordComment("[  Add64(t0, a0, Operand(zero_reg));");
    __ Add64(t0, a0, Operand(zero_reg));
    __ RecordComment("]");

    __ RecordComment("[  bind(&copy);");
    __ bind(&copy);
    __ RecordComment("]");
    __ RecordComment("[  Ld(t1, MemOperand(src, 0));");
    __ Ld(t1, MemOperand(src, 0));
    __ RecordComment("]");
    __ RecordComment("[  Sd(t1, MemOperand(dest, 0));");
    __ Sd(t1, MemOperand(dest, 0));
    __ RecordComment("]");
    __ RecordComment("[  Sub64(t0, t0, Operand(8));");
    __ Sub64(t0, t0, Operand(8));
    __ RecordComment("]");
    __ RecordComment("[  Add64(src, src, Operand(kTaggedSize));");
    __ Add64(src, src, Operand(kTaggedSize));
    __ RecordComment("]");
    __ RecordComment("[  Add64(dest, dest, Operand(kTaggedSize));");
    __ Add64(dest, dest, Operand(kTaggedSize));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&copy, ge, t0, Operand(zero_reg));");
    __ Branch(&copy, ge, t0, Operand(zero_reg));
    __ RecordComment("]");
  }

  // Push arguments onto the stack (thisArgument is already on the stack).
  {
    Label done, push, loop;
    Register src = a6;
    Register scratch = len;
    __ RecordComment(
        "[  Add64(src, args, FixedArray::kHeaderSize - kHeapObjectTag);");
    __ Add64(src, args, FixedArray::kHeaderSize - kHeapObjectTag);
    __ RecordComment("]");
    __ RecordComment(
        "[  Add64(a0, a0, len);  // The 'len' argument for Call() or "
        "Construct().");
    __ Add64(a0, a0, len);  // The 'len' argument for Call() or Construct().
    __ RecordComment("]");
    __ RecordComment("[  Branch(&done, eq, len, Operand(zero_reg));");
    __ Branch(&done, eq, len, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment("[  Sll64(scratch, len, kTaggedSizeLog2);");
    __ Sll64(scratch, len, kTaggedSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(scratch, sp, Operand(scratch));");
    __ Sub64(scratch, sp, Operand(scratch));
    __ RecordComment("]");
    __ RecordComment("[  LoadRoot(t1, RootIndex::kTheHoleValue);");
    __ LoadRoot(t1, RootIndex::kTheHoleValue);
    __ RecordComment("]");
    __ RecordComment("[  bind(&loop);");
    __ bind(&loop);
    __ RecordComment("]");
    __ RecordComment("[  LoadAnyTaggedField(a5, MemOperand(src));");
    __ LoadAnyTaggedField(a5, MemOperand(src));
    __ RecordComment("]");
    __ RecordComment("[  Add64(src, src, kTaggedSize);");
    __ Add64(src, src, kTaggedSize);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&push, ne, a5, Operand(t1));");
    __ Branch(&push, ne, a5, Operand(t1));
    __ RecordComment("]");
    __ RecordComment("[  LoadRoot(a5, RootIndex::kUndefinedValue);");
    __ LoadRoot(a5, RootIndex::kUndefinedValue);
    __ RecordComment("]");
    __ RecordComment("[  bind(&push);");
    __ bind(&push);
    __ RecordComment("]");
    __ RecordComment("[  Sd(a5, MemOperand(a7, 0));");
    __ Sd(a5, MemOperand(a7, 0));
    __ RecordComment("]");
    __ RecordComment("[  Add64(a7, a7, Operand(kTaggedSize));");
    __ Add64(a7, a7, Operand(kTaggedSize));
    __ RecordComment("]");
    __ RecordComment("[  Add64(scratch, scratch, Operand(kTaggedSize));");
    __ Add64(scratch, scratch, Operand(kTaggedSize));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&loop, ne, scratch, Operand(sp));");
    __ Branch(&loop, ne, scratch, Operand(sp));
    __ RecordComment("]");
    __ RecordComment("[  bind(&done);");
    __ bind(&done);
    __ RecordComment("]");
  }

  // Tail-call to the actual Call or Construct builtin.
  __ RecordComment("[  Jump(code, RelocInfo::CODE_TARGET);");
  __ Jump(code, RelocInfo::CODE_TARGET);
  __ RecordComment("]");

  __ RecordComment("[  bind(&stack_overflow);");
  __ bind(&stack_overflow);
  __ RecordComment("]");
  __ RecordComment("[  TailCallRuntime(Runtime::kThrowStackOverflow);");
  __ TailCallRuntime(Runtime::kThrowStackOverflow);
  __ RecordComment("]");
}

// static
void Builtins::Generate_CallOrConstructForwardVarargs(MacroAssembler* masm,
                                                      CallOrConstructMode mode,
                                                      Handle<Code> code) {
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a3 : the new.target (for [[Construct]] calls)
  //  -- a1 : the target to call (can be any Object)
  //  -- a2 : start index (to support rest parameters)
  // -----------------------------------

  // Check if new.target has a [[Construct]] internal method.
  if (mode == CallOrConstructMode::kConstruct) {
    Label new_target_constructor, new_target_not_constructor;
    __ RecordComment("[  JumpIfSmi(a3, &new_target_not_constructor);");
    __ JumpIfSmi(a3, &new_target_not_constructor);
    __ RecordComment("]");
    __ RecordComment(
        "[  LoadTaggedPointerField(t1, FieldMemOperand(a3, "
        "HeapObject::kMapOffset));");
    __ LoadTaggedPointerField(t1, FieldMemOperand(a3, HeapObject::kMapOffset));
    __ RecordComment("]");
    __ RecordComment("[  Lbu(t1, FieldMemOperand(t1, Map::kBitFieldOffset));");
    __ Lbu(t1, FieldMemOperand(t1, Map::kBitFieldOffset));
    __ RecordComment("]");
    __ RecordComment(
        "[  And(t1, t1, Operand(Map::Bits1::IsConstructorBit::kMask));");
    __ And(t1, t1, Operand(Map::Bits1::IsConstructorBit::kMask));
    __ RecordComment("]");
    __ RecordComment(
        "[  Branch(&new_target_constructor, ne, t1, Operand(zero_reg));");
    __ Branch(&new_target_constructor, ne, t1, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment("[  bind(&new_target_not_constructor);");
    __ bind(&new_target_not_constructor);
    __ RecordComment("]");
    {
      FrameScope scope(masm, StackFrame::MANUAL);
      __ RecordComment("[  EnterFrame(StackFrame::INTERNAL);");
      __ EnterFrame(StackFrame::INTERNAL);
      __ RecordComment("]");
      __ RecordComment("[  Push(a3);");
      __ Push(a3);
      __ RecordComment("]");
      __ RecordComment("[  CallRuntime(Runtime::kThrowNotConstructor);");
      __ CallRuntime(Runtime::kThrowNotConstructor);
      __ RecordComment("]");
    }
    __ RecordComment("[  bind(&new_target_constructor);");
    __ bind(&new_target_constructor);
    __ RecordComment("]");
  }

  // Check if we have an arguments adaptor frame below the function frame.
  Label arguments_adaptor, arguments_done;
  __ RecordComment(
      "[  Ld(a6, MemOperand(fp, StandardFrameConstants::kCallerFPOffset));");
  __ Ld(a6, MemOperand(fp, StandardFrameConstants::kCallerFPOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Ld(a7, MemOperand(a6, "
      "CommonFrameConstants::kContextOrFrameTypeOffset));");
  __ Ld(a7, MemOperand(a6, CommonFrameConstants::kContextOrFrameTypeOffset));
  __ RecordComment("]");
  __ Branch(&arguments_adaptor, eq, a7,
            Operand(StackFrame::TypeToMarker(StackFrame::ARGUMENTS_ADAPTOR)));
  {
    __ RecordComment(
        "[  Ld(a7, MemOperand(fp, StandardFrameConstants::kFunctionOffset));");
    __ Ld(a7, MemOperand(fp, StandardFrameConstants::kFunctionOffset));
    __ RecordComment("]");
    __ RecordComment(
        "[  LoadTaggedPointerField(a7, FieldMemOperand(a7, "
        "JSFunction::kSharedFunctionInfoOffset));");
    __ LoadTaggedPointerField(
        a7, FieldMemOperand(a7, JSFunction::kSharedFunctionInfoOffset));
    __ RecordComment("]");
    __ Lhu(a7, FieldMemOperand(
                   a7, SharedFunctionInfo::kFormalParameterCountOffset));
    __ RecordComment("[  Move(a6, fp);");
    __ Move(a6, fp);
    __ RecordComment("]");
  }
  __ RecordComment("[  Branch(&arguments_done);");
  __ Branch(&arguments_done);
  __ RecordComment("]");
  __ RecordComment("[  bind(&arguments_adaptor);");
  __ bind(&arguments_adaptor);
  __ RecordComment("]");
  {
    // Just get the length from the ArgumentsAdaptorFrame.
    __ SmiUntag(a7,
                MemOperand(a6, ArgumentsAdaptorFrameConstants::kLengthOffset));
  }
  __ RecordComment("[  bind(&arguments_done);");
  __ bind(&arguments_done);
  __ RecordComment("]");

  Label stack_done, stack_overflow;
  __ RecordComment("[  Sub32(a7, a7, a2);");
  __ Sub32(a7, a7, a2);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&stack_done, le, a7, Operand(zero_reg));");
  __ Branch(&stack_done, le, a7, Operand(zero_reg));
  __ RecordComment("]");
  {
    // Check for stack overflow.
    Generate_StackOverflowCheck(masm, a7, a4, a5, &stack_overflow);

    // Forward the arguments from the caller frame.

    // Point to the first argument to copy (skipping the receiver).
    __ Add64(a6, a6,
             Operand(CommonFrameConstants::kFixedFrameSizeAboveFp +
                     kSystemPointerSize));
    __ RecordComment(
        "[  CalcScaledAddress(a6, a6, a2, kSystemPointerSizeLog2);");
    __ CalcScaledAddress(a6, a6, a2, kSystemPointerSizeLog2);
    __ RecordComment("]");

    // Move the arguments already in the stack,
    // including the receiver and the return address.
    {
      Label copy;
      Register src = t0, dest = a2;
      __ RecordComment("[  Move(src, sp);");
      __ Move(src, sp);
      __ RecordComment("]");
      // Update stack pointer.
      __ RecordComment("[  Sll64(t1, a7, kSystemPointerSizeLog2);");
      __ Sll64(t1, a7, kSystemPointerSizeLog2);
      __ RecordComment("]");
      __ RecordComment("[  Sub64(sp, sp, Operand(t1));");
      __ Sub64(sp, sp, Operand(t1));
      __ RecordComment("]");
      __ RecordComment("[  Move(dest, sp);");
      __ Move(dest, sp);
      __ RecordComment("]");
      __ RecordComment("[  Add64(t2, a0, Operand(zero_reg));");
      __ Add64(t2, a0, Operand(zero_reg));
      __ RecordComment("]");

      __ RecordComment("[  bind(&copy);");
      __ bind(&copy);
      __ RecordComment("]");
      __ RecordComment("[  Ld(t1, MemOperand(src, 0));");
      __ Ld(t1, MemOperand(src, 0));
      __ RecordComment("]");
      __ RecordComment("[  Sd(t1, MemOperand(dest, 0));");
      __ Sd(t1, MemOperand(dest, 0));
      __ RecordComment("]");
      __ RecordComment("[  Sub64(t2, t2, Operand(1));");
      __ Sub64(t2, t2, Operand(1));
      __ RecordComment("]");
      __ RecordComment("[  Add64(src, src, Operand(kSystemPointerSize));");
      __ Add64(src, src, Operand(kSystemPointerSize));
      __ RecordComment("]");
      __ RecordComment("[  Add64(dest, dest, Operand(kSystemPointerSize));");
      __ Add64(dest, dest, Operand(kSystemPointerSize));
      __ RecordComment("]");
      __ RecordComment("[  Branch(&copy, ge, t2, Operand(zero_reg));");
      __ Branch(&copy, ge, t2, Operand(zero_reg));
      __ RecordComment("]");
    }

    // Copy arguments from the caller frame.
    // TODO(victorgomes): Consider using forward order as potentially more cache
    // friendly.
    {
      Label loop;
      __ RecordComment("[  Add64(a0, a0, a7);");
      __ Add64(a0, a0, a7);
      __ RecordComment("]");
      __ RecordComment("[  bind(&loop);");
      __ bind(&loop);
      __ RecordComment("]");
      {
        __ RecordComment("[  Sub32(a7, a7, Operand(1));");
        __ Sub32(a7, a7, Operand(1));
        __ RecordComment("]");
        __ RecordComment(
            "[  CalcScaledAddress(t0, a6, a7, kSystemPointerSizeLog2);");
        __ CalcScaledAddress(t0, a6, a7, kSystemPointerSizeLog2);
        __ RecordComment("]");
        __ RecordComment("[  Ld(kScratchReg, MemOperand(t0));");
        __ Ld(kScratchReg, MemOperand(t0));
        __ RecordComment("]");
        __ RecordComment(
            "[  CalcScaledAddress(t0, a2, a7, kSystemPointerSizeLog2);");
        __ CalcScaledAddress(t0, a2, a7, kSystemPointerSizeLog2);
        __ RecordComment("]");
        __ RecordComment("[  Sd(kScratchReg, MemOperand(t0));");
        __ Sd(kScratchReg, MemOperand(t0));
        __ RecordComment("]");
        __ RecordComment("[  Branch(&loop, ne, a7, Operand(zero_reg));");
        __ Branch(&loop, ne, a7, Operand(zero_reg));
        __ RecordComment("]");
      }
    }
  }
  __ RecordComment("[  Branch(&stack_done);");
  __ Branch(&stack_done);
  __ RecordComment("]");
  __ RecordComment("[  bind(&stack_overflow);");
  __ bind(&stack_overflow);
  __ RecordComment("]");
  __ RecordComment("[  TailCallRuntime(Runtime::kThrowStackOverflow);");
  __ TailCallRuntime(Runtime::kThrowStackOverflow);
  __ RecordComment("]");
  __ RecordComment("[  bind(&stack_done);");
  __ bind(&stack_done);
  __ RecordComment("]");

  // Tail-call to the {code} handler.
  __ RecordComment("[  Jump(code, RelocInfo::CODE_TARGET);");
  __ Jump(code, RelocInfo::CODE_TARGET);
  __ RecordComment("]");
}

// static
void Builtins::Generate_CallFunction(MacroAssembler* masm,
                                     ConvertReceiverMode mode) {
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the function to call (checked to be a JSFunction)
  // -----------------------------------
  __ RecordComment("[  AssertFunction(a1);");
  __ AssertFunction(a1);
  __ RecordComment("]");

  // See ES6 section 9.2.1 [[Call]] ( thisArgument, argumentsList)
  // Check that function is not a "classConstructor".
  Label class_constructor;
  __ RecordComment(
      "[  LoadTaggedPointerField(a2, FieldMemOperand(a1, "
      "JSFunction::kSharedFunctionInfoOffset));");
  __ LoadTaggedPointerField(
      a2, FieldMemOperand(a1, JSFunction::kSharedFunctionInfoOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Lwu(a3, FieldMemOperand(a2, SharedFunctionInfo::kFlagsOffset));");
  __ Lwu(a3, FieldMemOperand(a2, SharedFunctionInfo::kFlagsOffset));
  __ RecordComment("]");
  __ And(kScratchReg, a3,
         Operand(SharedFunctionInfo::IsClassConstructorBit::kMask));
  __ RecordComment(
      "[  Branch(&class_constructor, ne, kScratchReg, Operand(zero_reg));");
  __ Branch(&class_constructor, ne, kScratchReg, Operand(zero_reg));
  __ RecordComment("]");

  // Enter the context of the function; ToObject has to run in the function
  // context, and we also need to take the global proxy from the function
  // context in case of conversion.
  __ RecordComment(
      "[  LoadTaggedPointerField(cp, FieldMemOperand(a1, "
      "JSFunction::kContextOffset));");
  __ LoadTaggedPointerField(cp,
                            FieldMemOperand(a1, JSFunction::kContextOffset));
  __ RecordComment("]");
  // We need to convert the receiver for non-native sloppy mode functions.
  Label done_convert;
  __ RecordComment(
      "[  Lwu(a3, FieldMemOperand(a2, SharedFunctionInfo::kFlagsOffset));");
  __ Lwu(a3, FieldMemOperand(a2, SharedFunctionInfo::kFlagsOffset));
  __ RecordComment("]");
  __ And(kScratchReg, a3,
         Operand(SharedFunctionInfo::IsNativeBit::kMask |
                 SharedFunctionInfo::IsStrictBit::kMask));
  __ RecordComment(
      "[  Branch(&done_convert, ne, kScratchReg, Operand(zero_reg));");
  __ Branch(&done_convert, ne, kScratchReg, Operand(zero_reg));
  __ RecordComment("]");
  {
    // ----------- S t a t e -------------
    //  -- a0 : the number of arguments (not including the receiver)
    //  -- a1 : the function to call (checked to be a JSFunction)
    //  -- a2 : the shared function info.
    //  -- cp : the function context.
    // -----------------------------------

    if (mode == ConvertReceiverMode::kNullOrUndefined) {
      // Patch receiver to global proxy.
      __ RecordComment("[  LoadGlobalProxy(a3);");
      __ LoadGlobalProxy(a3);
      __ RecordComment("]");
    } else {
      Label convert_to_object, convert_receiver;
      __ RecordComment("[  LoadReceiver(a3, a0);");
      __ LoadReceiver(a3, a0);
      __ RecordComment("]");
      __ RecordComment("[  JumpIfSmi(a3, &convert_to_object);");
      __ JumpIfSmi(a3, &convert_to_object);
      __ RecordComment("]");
      STATIC_ASSERT(LAST_JS_RECEIVER_TYPE == LAST_TYPE);
      __ RecordComment("[  GetObjectType(a3, a4, a4);");
      __ GetObjectType(a3, a4, a4);
      __ RecordComment("]");
      __ Branch(&done_convert, Ugreater_equal, a4,
                Operand(FIRST_JS_RECEIVER_TYPE));
      if (mode != ConvertReceiverMode::kNotNullOrUndefined) {
        Label convert_global_proxy;
        __ RecordComment(
            "[  JumpIfRoot(a3, RootIndex::kUndefinedValue, "
            "&convert_global_proxy);");
        __ JumpIfRoot(a3, RootIndex::kUndefinedValue, &convert_global_proxy);
        __ RecordComment("]");
        __ RecordComment(
            "[  JumpIfNotRoot(a3, RootIndex::kNullValue, &convert_to_object);");
        __ JumpIfNotRoot(a3, RootIndex::kNullValue, &convert_to_object);
        __ RecordComment("]");
        __ RecordComment("[  bind(&convert_global_proxy);");
        __ bind(&convert_global_proxy);
        __ RecordComment("]");
        {
          // Patch receiver to global proxy.
          __ RecordComment("[  LoadGlobalProxy(a3);");
          __ LoadGlobalProxy(a3);
          __ RecordComment("]");
        }
        __ RecordComment("[  Branch(&convert_receiver);");
        __ Branch(&convert_receiver);
        __ RecordComment("]");
      }
      __ RecordComment("[  bind(&convert_to_object);");
      __ bind(&convert_to_object);
      __ RecordComment("]");
      {
        // Convert receiver using ToObject.
        // TODO(bmeurer): Inline the allocation here to avoid building the frame
        // in the fast case? (fall back to AllocateInNewSpace?)
        FrameScope scope(masm, StackFrame::INTERNAL);
        __ RecordComment("[  SmiTag(a0);");
        __ SmiTag(a0);
        __ RecordComment("]");
        __ RecordComment("[  Push(a0, a1);");
        __ Push(a0, a1);
        __ RecordComment("]");
        __ RecordComment("[  Move(a0, a3);");
        __ Move(a0, a3);
        __ RecordComment("]");
        __ RecordComment("[  Push(cp);");
        __ Push(cp);
        __ RecordComment("]");
        __ Call(BUILTIN_CODE(masm->isolate(), ToObject),
                RelocInfo::CODE_TARGET);
        __ RecordComment("[  Pop(cp);");
        __ Pop(cp);
        __ RecordComment("]");
        __ RecordComment("[  Move(a3, a0);");
        __ Move(a3, a0);
        __ RecordComment("]");
        __ RecordComment("[  Pop(a0, a1);");
        __ Pop(a0, a1);
        __ RecordComment("]");
        __ RecordComment("[  SmiUntag(a0);");
        __ SmiUntag(a0);
        __ RecordComment("]");
      }
      __ RecordComment(
          "[  LoadTaggedPointerField(a2, FieldMemOperand(a1, "
          "JSFunction::kSharedFunctionInfoOffset));");
      __ LoadTaggedPointerField(
          a2, FieldMemOperand(a1, JSFunction::kSharedFunctionInfoOffset));
      __ RecordComment("]");
      __ RecordComment("[  bind(&convert_receiver);");
      __ bind(&convert_receiver);
      __ RecordComment("]");
    }
    __ RecordComment("[  StoreReceiver(a3, a0, kScratchReg);");
    __ StoreReceiver(a3, a0, kScratchReg);
    __ RecordComment("]");
  }
  __ RecordComment("[  bind(&done_convert);");
  __ bind(&done_convert);
  __ RecordComment("]");

  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the function to call (checked to be a JSFunction)
  //  -- a2 : the shared function info.
  //  -- cp : the function context.
  // -----------------------------------

  __ Lhu(a2,
         FieldMemOperand(a2, SharedFunctionInfo::kFormalParameterCountOffset));
  __ RecordComment("[  InvokeFunctionCode(a1, no_reg, a2, a0, JUMP_FUNCTION);");
  __ InvokeFunctionCode(a1, no_reg, a2, a0, JUMP_FUNCTION);
  __ RecordComment("]");

  // The function is a "classConstructor", need to raise an exception.
  __ RecordComment("[  bind(&class_constructor);");
  __ bind(&class_constructor);
  __ RecordComment("]");
  {
    FrameScope frame(masm, StackFrame::INTERNAL);
    __ RecordComment("[  Push(a1);");
    __ Push(a1);
    __ RecordComment("]");
    __ RecordComment(
        "[  CallRuntime(Runtime::kThrowConstructorNonCallableError);");
    __ CallRuntime(Runtime::kThrowConstructorNonCallableError);
    __ RecordComment("]");
  }
}

// static
void Builtins::Generate_CallBoundFunctionImpl(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the function to call (checked to be a JSBoundFunction)
  // -----------------------------------
  __ RecordComment("[  AssertBoundFunction(a1);");
  __ AssertBoundFunction(a1);
  __ RecordComment("]");

  // Patch the receiver to [[BoundThis]].
  {
    __ RecordComment(
        "[  LoadAnyTaggedField(t0, FieldMemOperand(a1, "
        "JSBoundFunction::kBoundThisOffset));");
    __ LoadAnyTaggedField(
        t0, FieldMemOperand(a1, JSBoundFunction::kBoundThisOffset));
    __ RecordComment("]");
    __ RecordComment("[  StoreReceiver(t0, a0, kScratchReg);");
    __ StoreReceiver(t0, a0, kScratchReg);
    __ RecordComment("]");
  }

  // Load [[BoundArguments]] into a2 and length of that into a4.
  __ RecordComment(
      "[  LoadTaggedPointerField(a2, FieldMemOperand(a1, "
      "JSBoundFunction::kBoundArgumentsOffset));");
  __ LoadTaggedPointerField(
      a2, FieldMemOperand(a1, JSBoundFunction::kBoundArgumentsOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  SmiUntagField(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));");
  __ SmiUntagField(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));
  __ RecordComment("]");

  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the function to call (checked to be a JSBoundFunction)
  //  -- a2 : the [[BoundArguments]] (implemented as FixedArray)
  //  -- a4 : the number of [[BoundArguments]]
  // -----------------------------------

  // Reserve stack space for the [[BoundArguments]].
  {
    Label done;
    __ RecordComment("[  Sll64(a5, a4, kSystemPointerSizeLog2);");
    __ Sll64(a5, a4, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(t0, sp, Operand(a5));");
    __ Sub64(t0, sp, Operand(a5));
    __ RecordComment("]");
    // Check the stack for overflow. We are not trying to catch interruptions
    // (i.e. debug break and preemption) here, so check the "real stack limit".
    LoadStackLimit(masm, kScratchReg, StackLimitKind::kRealStackLimit);
    __ RecordComment(
        "[  Branch(&done, Ugreater_equal, t0, Operand(kScratchReg));");
    __ Branch(&done, Ugreater_equal, t0, Operand(kScratchReg));
    __ RecordComment("]");
    {
      FrameScope scope(masm, StackFrame::MANUAL);
      __ RecordComment("[  EnterFrame(StackFrame::INTERNAL);");
      __ EnterFrame(StackFrame::INTERNAL);
      __ RecordComment("]");
      __ RecordComment("[  CallRuntime(Runtime::kThrowStackOverflow);");
      __ CallRuntime(Runtime::kThrowStackOverflow);
      __ RecordComment("]");
    }
    __ RecordComment("[  bind(&done);");
    __ bind(&done);
    __ RecordComment("]");
  }

  // Pop receiver.
  __ RecordComment("[  Pop(t0);");
  __ Pop(t0);
  __ RecordComment("]");

  // Copy [[BoundArguments]].
  {
    Label loop, done_loop;
    __ RecordComment(
        "[  SmiUntag(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));");
    __ SmiUntag(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));
    __ RecordComment("]");
    __ RecordComment("[  Add64(a0, a0, Operand(a4));");
    __ Add64(a0, a0, Operand(a4));
    __ RecordComment("]");
    __ RecordComment(
        "[  Add64(a2, a2, Operand(FixedArray::kHeaderSize - kHeapObjectTag));");
    __ Add64(a2, a2, Operand(FixedArray::kHeaderSize - kHeapObjectTag));
    __ RecordComment("]");
    __ RecordComment("[  bind(&loop);");
    __ bind(&loop);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(a4, a4, Operand(1));");
    __ Sub64(a4, a4, Operand(1));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&done_loop, lt, a4, Operand(zero_reg));");
    __ Branch(&done_loop, lt, a4, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment(
        "[  CalcScaledAddress(a5, a2, a4, kSystemPointerSizeLog2);");
    __ CalcScaledAddress(a5, a2, a4, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Ld(kScratchReg, MemOperand(a5));");
    __ Ld(kScratchReg, MemOperand(a5));
    __ RecordComment("]");
    __ RecordComment("[  Push(kScratchReg);");
    __ Push(kScratchReg);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&loop);");
    __ Branch(&loop);
    __ RecordComment("]");
    __ RecordComment("[  bind(&done_loop);");
    __ bind(&done_loop);
    __ RecordComment("]");
  }

  // Push receiver.
  __ RecordComment("[  Push(t0);");
  __ Push(t0);
  __ RecordComment("]");

  // Call the [[BoundTargetFunction]] via the Call builtin.
  __ RecordComment(
      "[  Ld(a1, FieldMemOperand(a1, "
      "JSBoundFunction::kBoundTargetFunctionOffset));");
  __ Ld(a1, FieldMemOperand(a1, JSBoundFunction::kBoundTargetFunctionOffset));
  __ RecordComment("]");
  __ Jump(BUILTIN_CODE(masm->isolate(), Call_ReceiverIsAny),
          RelocInfo::CODE_TARGET);
}

// static
void Builtins::Generate_Call(MacroAssembler* masm, ConvertReceiverMode mode) {
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the target to call (can be any Object).
  // -----------------------------------

  Label non_callable, non_smi;
  __ RecordComment("[  JumpIfSmi(a1, &non_callable);");
  __ JumpIfSmi(a1, &non_callable);
  __ RecordComment("]");
  __ RecordComment("[  bind(&non_smi);");
  __ bind(&non_smi);
  __ RecordComment("]");
  __ RecordComment("[  GetObjectType(a1, t1, t2);");
  __ GetObjectType(a1, t1, t2);
  __ RecordComment("]");
  __ Jump(masm->isolate()->builtins()->CallFunction(mode),
          RelocInfo::CODE_TARGET, eq, t2, Operand(JS_FUNCTION_TYPE));
  __ Jump(BUILTIN_CODE(masm->isolate(), CallBoundFunction),
          RelocInfo::CODE_TARGET, eq, t2, Operand(JS_BOUND_FUNCTION_TYPE));

  // Check if target has a [[Call]] internal method.
  __ RecordComment("[  Lbu(t1, FieldMemOperand(t1, Map::kBitFieldOffset));");
  __ Lbu(t1, FieldMemOperand(t1, Map::kBitFieldOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  And(t1, t1, Operand(Map::Bits1::IsCallableBit::kMask));");
  __ And(t1, t1, Operand(Map::Bits1::IsCallableBit::kMask));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&non_callable, eq, t1, Operand(zero_reg));");
  __ Branch(&non_callable, eq, t1, Operand(zero_reg));
  __ RecordComment("]");

  __ Jump(BUILTIN_CODE(masm->isolate(), CallProxy), RelocInfo::CODE_TARGET, eq,
          t2, Operand(JS_PROXY_TYPE));

  // 2. Call to something else, which might have a [[Call]] internal method (if
  // not we raise an exception).
  // Overwrite the original receiver with the (original) target.
  __ RecordComment("[  StoreReceiver(a1, a0, kScratchReg);");
  __ StoreReceiver(a1, a0, kScratchReg);
  __ RecordComment("]");
  // Let the "call_as_function_delegate" take care of the rest.
  __ RecordComment(
      "[  LoadNativeContextSlot(Context::CALL_AS_FUNCTION_DELEGATE_INDEX, "
      "a1);");
  __ LoadNativeContextSlot(Context::CALL_AS_FUNCTION_DELEGATE_INDEX, a1);
  __ RecordComment("]");
  __ Jump(masm->isolate()->builtins()->CallFunction(
              ConvertReceiverMode::kNotNullOrUndefined),
          RelocInfo::CODE_TARGET);

  // 3. Call to something that is not callable.
  __ RecordComment("[  bind(&non_callable);");
  __ bind(&non_callable);
  __ RecordComment("]");
  {
    FrameScope scope(masm, StackFrame::INTERNAL);
    __ RecordComment("[  Push(a1);");
    __ Push(a1);
    __ RecordComment("]");
    __ RecordComment("[  CallRuntime(Runtime::kThrowCalledNonCallable);");
    __ CallRuntime(Runtime::kThrowCalledNonCallable);
    __ RecordComment("]");
  }
}

void Builtins::Generate_ConstructFunction(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the constructor to call (checked to be a JSFunction)
  //  -- a3 : the new target (checked to be a constructor)
  // -----------------------------------
  __ RecordComment("[  AssertConstructor(a1);");
  __ AssertConstructor(a1);
  __ RecordComment("]");
  __ RecordComment("[  AssertFunction(a1);");
  __ AssertFunction(a1);
  __ RecordComment("]");

  // Calling convention for function specific ConstructStubs require
  // a2 to contain either an AllocationSite or undefined.
  __ RecordComment("[  LoadRoot(a2, RootIndex::kUndefinedValue);");
  __ LoadRoot(a2, RootIndex::kUndefinedValue);
  __ RecordComment("]");

  Label call_generic_stub;

  // Jump to JSBuiltinsConstructStub or JSConstructStubGeneric.
  __ RecordComment(
      "[  LoadTaggedPointerField(a4, FieldMemOperand(a1, "
      "JSFunction::kSharedFunctionInfoOffset));");
  __ LoadTaggedPointerField(
      a4, FieldMemOperand(a1, JSFunction::kSharedFunctionInfoOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Lwu(a4, FieldMemOperand(a4, SharedFunctionInfo::kFlagsOffset));");
  __ Lwu(a4, FieldMemOperand(a4, SharedFunctionInfo::kFlagsOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  And(a4, a4, "
      "Operand(SharedFunctionInfo::ConstructAsBuiltinBit::kMask));");
  __ And(a4, a4, Operand(SharedFunctionInfo::ConstructAsBuiltinBit::kMask));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&call_generic_stub, eq, a4, Operand(zero_reg));");
  __ Branch(&call_generic_stub, eq, a4, Operand(zero_reg));
  __ RecordComment("]");

  __ Jump(BUILTIN_CODE(masm->isolate(), JSBuiltinsConstructStub),
          RelocInfo::CODE_TARGET);

  __ RecordComment("[  bind(&call_generic_stub);");
  __ bind(&call_generic_stub);
  __ RecordComment("]");
  __ Jump(BUILTIN_CODE(masm->isolate(), JSConstructStubGeneric),
          RelocInfo::CODE_TARGET);
}

// static
void Builtins::Generate_ConstructBoundFunction(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the function to call (checked to be a JSBoundFunction)
  //  -- a3 : the new target (checked to be a constructor)
  // -----------------------------------
  __ RecordComment("[  AssertConstructor(a1);");
  __ AssertConstructor(a1);
  __ RecordComment("]");
  __ RecordComment("[  AssertBoundFunction(a1);");
  __ AssertBoundFunction(a1);
  __ RecordComment("]");

  // Load [[BoundArguments]] into a2 and length of that into a4.
  __ RecordComment(
      "[  LoadTaggedPointerField(a2, FieldMemOperand(a1, "
      "JSBoundFunction::kBoundArgumentsOffset));");
  __ LoadTaggedPointerField(
      a2, FieldMemOperand(a1, JSBoundFunction::kBoundArgumentsOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  SmiUntagField(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));");
  __ SmiUntagField(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));
  __ RecordComment("]");

  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the function to call (checked to be a JSBoundFunction)
  //  -- a2 : the [[BoundArguments]] (implemented as FixedArray)
  //  -- a3 : the new target (checked to be a constructor)
  //  -- a4 : the number of [[BoundArguments]]
  // -----------------------------------

  // Reserve stack space for the [[BoundArguments]].
  {
    Label done;
    __ RecordComment("[  Sll64(a5, a4, kSystemPointerSizeLog2);");
    __ Sll64(a5, a4, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(t0, sp, Operand(a5));");
    __ Sub64(t0, sp, Operand(a5));
    __ RecordComment("]");
    // Check the stack for overflow. We are not trying to catch interruptions
    // (i.e. debug break and preemption) here, so check the "real stack limit".
    LoadStackLimit(masm, kScratchReg, StackLimitKind::kRealStackLimit);
    __ RecordComment(
        "[  Branch(&done, Ugreater_equal, t0, Operand(kScratchReg));");
    __ Branch(&done, Ugreater_equal, t0, Operand(kScratchReg));
    __ RecordComment("]");
    {
      FrameScope scope(masm, StackFrame::MANUAL);
      __ RecordComment("[  EnterFrame(StackFrame::INTERNAL);");
      __ EnterFrame(StackFrame::INTERNAL);
      __ RecordComment("]");
      __ RecordComment("[  CallRuntime(Runtime::kThrowStackOverflow);");
      __ CallRuntime(Runtime::kThrowStackOverflow);
      __ RecordComment("]");
    }
    __ RecordComment("[  bind(&done);");
    __ bind(&done);
    __ RecordComment("]");
  }

  // Pop receiver.
  __ RecordComment("[  Pop(t0);");
  __ Pop(t0);
  __ RecordComment("]");

  // Push [[BoundArguments]].
  {
    Label loop, done_loop;
    __ RecordComment(
        "[  SmiUntag(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));");
    __ SmiUntag(a4, FieldMemOperand(a2, FixedArray::kLengthOffset));
    __ RecordComment("]");
    __ RecordComment("[  Add64(a0, a0, Operand(a4));");
    __ Add64(a0, a0, Operand(a4));
    __ RecordComment("]");
    __ RecordComment(
        "[  Add64(a2, a2, Operand(FixedArray::kHeaderSize - kHeapObjectTag));");
    __ Add64(a2, a2, Operand(FixedArray::kHeaderSize - kHeapObjectTag));
    __ RecordComment("]");
    __ RecordComment("[  bind(&loop);");
    __ bind(&loop);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(a4, a4, Operand(1));");
    __ Sub64(a4, a4, Operand(1));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&done_loop, lt, a4, Operand(zero_reg));");
    __ Branch(&done_loop, lt, a4, Operand(zero_reg));
    __ RecordComment("]");
    __ RecordComment(
        "[  CalcScaledAddress(a5, a2, a4, kSystemPointerSizeLog2);");
    __ CalcScaledAddress(a5, a2, a4, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Ld(kScratchReg, MemOperand(a5));");
    __ Ld(kScratchReg, MemOperand(a5));
    __ RecordComment("]");
    __ RecordComment("[  Push(kScratchReg);");
    __ Push(kScratchReg);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&loop);");
    __ Branch(&loop);
    __ RecordComment("]");
    __ RecordComment("[  bind(&done_loop);");
    __ bind(&done_loop);
    __ RecordComment("]");
  }

  // Push receiver.
  __ RecordComment("[  Push(t0);");
  __ Push(t0);
  __ RecordComment("]");

  // Patch new.target to [[BoundTargetFunction]] if new.target equals target.
  {
    Label skip_load;
    __ RecordComment("[  Branch(&skip_load, ne, a1, Operand(a3));");
    __ Branch(&skip_load, ne, a1, Operand(a3));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(a3, FieldMemOperand(a1, "
        "JSBoundFunction::kBoundTargetFunctionOffset));");
    __ Ld(a3, FieldMemOperand(a1, JSBoundFunction::kBoundTargetFunctionOffset));
    __ RecordComment("]");
    __ RecordComment("[  bind(&skip_load);");
    __ bind(&skip_load);
    __ RecordComment("]");
  }

  // Construct the [[BoundTargetFunction]] via the Construct builtin.
  __ RecordComment(
      "[  Ld(a1, FieldMemOperand(a1, "
      "JSBoundFunction::kBoundTargetFunctionOffset));");
  __ Ld(a1, FieldMemOperand(a1, JSBoundFunction::kBoundTargetFunctionOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Jump(BUILTIN_CODE(masm->isolate(), Construct), "
      "RelocInfo::CODE_TARGET);");
  __ Jump(BUILTIN_CODE(masm->isolate(), Construct), RelocInfo::CODE_TARGET);
  __ RecordComment("]");
}

// static
void Builtins::Generate_Construct(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- a0 : the number of arguments (not including the receiver)
  //  -- a1 : the constructor to call (can be any Object)
  //  -- a3 : the new target (either the same as the constructor or
  //          the JSFunction on which new was invoked initially)
  // -----------------------------------

  // Check if target is a Smi.
  Label non_constructor, non_proxy;
  __ RecordComment("[  JumpIfSmi(a1, &non_constructor);");
  __ JumpIfSmi(a1, &non_constructor);
  __ RecordComment("]");

  // Check if target has a [[Construct]] internal method.
  __ RecordComment(
      "[  LoadTaggedPointerField(t1, FieldMemOperand(a1, "
      "HeapObject::kMapOffset));");
  __ LoadTaggedPointerField(t1, FieldMemOperand(a1, HeapObject::kMapOffset));
  __ RecordComment("]");
  __ RecordComment("[  Lbu(t4, FieldMemOperand(t1, Map::kBitFieldOffset));");
  __ Lbu(t4, FieldMemOperand(t1, Map::kBitFieldOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  And(t4, t4, Operand(Map::Bits1::IsConstructorBit::kMask));");
  __ And(t4, t4, Operand(Map::Bits1::IsConstructorBit::kMask));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&non_constructor, eq, t4, Operand(zero_reg));");
  __ Branch(&non_constructor, eq, t4, Operand(zero_reg));
  __ RecordComment("]");

  // Dispatch based on instance type.
  __ RecordComment(
      "[  Lhu(t2, FieldMemOperand(t1, Map::kInstanceTypeOffset));");
  __ Lhu(t2, FieldMemOperand(t1, Map::kInstanceTypeOffset));
  __ RecordComment("]");
  __ Jump(BUILTIN_CODE(masm->isolate(), ConstructFunction),
          RelocInfo::CODE_TARGET, eq, t2, Operand(JS_FUNCTION_TYPE));

  // Only dispatch to bound functions after checking whether they are
  // constructors.
  __ Jump(BUILTIN_CODE(masm->isolate(), ConstructBoundFunction),
          RelocInfo::CODE_TARGET, eq, t2, Operand(JS_BOUND_FUNCTION_TYPE));

  // Only dispatch to proxies after checking whether they are constructors.
  __ RecordComment("[  Branch(&non_proxy, ne, t2, Operand(JS_PROXY_TYPE));");
  __ Branch(&non_proxy, ne, t2, Operand(JS_PROXY_TYPE));
  __ RecordComment("]");
  __ Jump(BUILTIN_CODE(masm->isolate(), ConstructProxy),
          RelocInfo::CODE_TARGET);

  // Called Construct on an exotic Object with a [[Construct]] internal method.
  __ RecordComment("[  bind(&non_proxy);");
  __ bind(&non_proxy);
  __ RecordComment("]");
  {
    // Overwrite the original receiver with the (original) target.
    __ RecordComment("[  StoreReceiver(a1, a0, kScratchReg);");
    __ StoreReceiver(a1, a0, kScratchReg);
    __ RecordComment("]");
    // Let the "call_as_constructor_delegate" take care of the rest.
    __ RecordComment(
        "[  LoadNativeContextSlot(Context::CALL_AS_CONSTRUCTOR_DELEGATE_INDEX, "
        "a1);");
    __ LoadNativeContextSlot(Context::CALL_AS_CONSTRUCTOR_DELEGATE_INDEX, a1);
    __ RecordComment("]");
    __ Jump(masm->isolate()->builtins()->CallFunction(),
            RelocInfo::CODE_TARGET);
  }

  // Called Construct on an Object that doesn't have a [[Construct]] internal
  // method.
  __ RecordComment("[  bind(&non_constructor);");
  __ bind(&non_constructor);
  __ RecordComment("]");
  __ Jump(BUILTIN_CODE(masm->isolate(), ConstructedNonConstructable),
          RelocInfo::CODE_TARGET);
}

void Builtins::Generate_ArgumentsAdaptorTrampoline(MacroAssembler* masm) {
  // State setup as expected by MacroAssembler::InvokePrologue.
  // ----------- S t a t e -------------
  //  -- a0: actual arguments count
  //  -- a1: function (passed through to callee)
  //  -- a2: expected arguments count
  //  -- a3: new target (passed through to callee)
  // -----------------------------------

  Label invoke, dont_adapt_arguments, stack_overflow;

  Label enough, too_few;
  __ Branch(&dont_adapt_arguments, eq, a2,
            Operand(kDontAdaptArgumentsSentinel));
  // We use Uless as the number of argument should always be greater than 0.
  __ RecordComment("[  Branch(&too_few, Uless, a0, Operand(a2));");
  __ Branch(&too_few, Uless, a0, Operand(a2));
  __ RecordComment("]");

  {  // Enough parameters: actual >= expected.
     // a0: actual number of arguments as a smi
     // a1: function
     // a2: expected number of arguments
     // a3: new target (passed through to callee)
    __ RecordComment("[  bind(&enough);");
    __ bind(&enough);
    __ RecordComment("]");
    EnterArgumentsAdaptorFrame(masm);
    Generate_StackOverflowCheck(masm, a2, a5, kScratchReg, &stack_overflow);

    // Calculate copy start address into a0 and copy end address into a4.
    __ RecordComment("[  Sll64(a0, a2, kSystemPointerSizeLog2);");
    __ Sll64(a0, a2, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Add64(a0, fp, a0);");
    __ Add64(a0, fp, a0);
    __ RecordComment("]");

    // Adjust for return address and receiver.
    __ RecordComment("[  Add64(a0, a0, Operand(2 * kSystemPointerSize));");
    __ Add64(a0, a0, Operand(2 * kSystemPointerSize));
    __ RecordComment("]");
    // Compute copy end address.
    __ RecordComment("[  Sll64(a4, a2, kSystemPointerSizeLog2);");
    __ Sll64(a4, a2, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(a4, a0, a4);");
    __ Sub64(a4, a0, a4);
    __ RecordComment("]");

    // Copy the arguments (including the receiver) to the new stack frame.
    // a0: copy start address
    // a1: function
    // a2: expected number of arguments
    // a3: new target (passed through to callee)
    // a4: copy end address

    Label copy;
    __ RecordComment("[  bind(&copy);");
    __ bind(&copy);
    __ RecordComment("]");
    __ RecordComment("[  Ld(a5, MemOperand(a0));");
    __ Ld(a5, MemOperand(a0));
    __ RecordComment("]");
    __ RecordComment("[  push(a5);");
    __ push(a5);
    __ RecordComment("]");
    __ RecordComment("[  Add64(a0, a0, -kSystemPointerSize);");
    __ Add64(a0, a0, -kSystemPointerSize);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&copy, ge, a0, Operand(a4));");
    __ Branch(&copy, ge, a0, Operand(a4));
    __ RecordComment("]");

    __ RecordComment("[  Branch(&invoke);");
    __ Branch(&invoke);
    __ RecordComment("]");
  }

  {  // Too few parameters: Actual < expected.
    __ RecordComment("[  bind(&too_few);");
    __ bind(&too_few);
    __ RecordComment("]");
    EnterArgumentsAdaptorFrame(masm);
    Generate_StackOverflowCheck(masm, a2, a5, kScratchReg, &stack_overflow);

    // Fill the remaining expected arguments with undefined.
    __ RecordComment("[  LoadRoot(t0, RootIndex::kUndefinedValue);");
    __ LoadRoot(t0, RootIndex::kUndefinedValue);
    __ RecordComment("]");
    __ RecordComment("[  SmiUntag(t1, a0);");
    __ SmiUntag(t1, a0);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(t2, a2, Operand(t1));");
    __ Sub64(t2, a2, Operand(t1));
    __ RecordComment("]");
    __ RecordComment("[  Sll64(a4, t2, kSystemPointerSizeLog2);");
    __ Sll64(a4, t2, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(a4, fp, a4);");
    __ Sub64(a4, fp, a4);
    __ RecordComment("]");
    // Adjust for frame.
    __ Sub64(a4, a4,
             Operand(ArgumentsAdaptorFrameConstants::kFixedFrameSizeFromFp +
                     kSystemPointerSize));

    Label fill;
    __ RecordComment("[  bind(&fill);");
    __ bind(&fill);
    __ RecordComment("]");
    __ RecordComment("[  push(t0);");
    __ push(t0);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&fill, ne, sp, Operand(a4));");
    __ Branch(&fill, ne, sp, Operand(a4));
    __ RecordComment("]");

    // Calculate copy start address into r0 and copy end address is fp.
    __ RecordComment("[  SmiScale(a0, a0, kSystemPointerSizeLog2);");
    __ SmiScale(a0, a0, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Add64(a0, fp, a0);");
    __ Add64(a0, fp, a0);
    __ RecordComment("]");

    // Copy the arguments (including the receiver) to the new stack frame.
    Label copy;
    __ RecordComment("[  bind(&copy);");
    __ bind(&copy);
    __ RecordComment("]");

    // Adjust load for return address and receiver.
    __ RecordComment("[  Ld(t0, MemOperand(a0, 2 * kSystemPointerSize));");
    __ Ld(t0, MemOperand(a0, 2 * kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment("[  push(t0);");
    __ push(t0);
    __ RecordComment("]");

    __ RecordComment("[  Sub64(a0, a0, Operand(kSystemPointerSize));");
    __ Sub64(a0, a0, Operand(kSystemPointerSize));
    __ RecordComment("]");
    __ RecordComment("[  Branch(&copy, ge, a0, Operand(fp));");
    __ Branch(&copy, ge, a0, Operand(fp));
    __ RecordComment("]");
  }

  // Call the entry point.
  __ RecordComment("[  bind(&invoke);");
  __ bind(&invoke);
  __ RecordComment("]");
  __ RecordComment("[  Move(a0, a2);");
  __ Move(a0, a2);
  __ RecordComment("]");
  // a0 : expected number of arguments
  // a1 : function (passed through to callee)
  // a3: new target (passed through to callee)
  static_assert(kJavaScriptCallCodeStartRegister == a2, "ABI mismatch");
  __ RecordComment(
      "[  LoadTaggedPointerField(a2, FieldMemOperand(a1, "
      "JSFunction::kCodeOffset));");
  __ LoadTaggedPointerField(a2, FieldMemOperand(a1, JSFunction::kCodeOffset));
  __ RecordComment("]");
  __ RecordComment("[  CallCodeObject(a2);");
  __ CallCodeObject(a2);
  __ RecordComment("]");

  // Store offset of return address for deoptimizer.
  masm->isolate()->heap()->SetArgumentsAdaptorDeoptPCOffset(masm->pc_offset());

  // Exit frame and return.
  LeaveArgumentsAdaptorFrame(masm);
  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");

  // -------------------------------------------
  // Don't adapt arguments.
  // -------------------------------------------
  __ RecordComment("[  bind(&dont_adapt_arguments);");
  __ bind(&dont_adapt_arguments);
  __ RecordComment("]");
  static_assert(kJavaScriptCallCodeStartRegister == a2, "ABI mismatch");
  __ RecordComment(
      "[  LoadTaggedPointerField(a2, FieldMemOperand(a1, "
      "JSFunction::kCodeOffset));");
  __ LoadTaggedPointerField(a2, FieldMemOperand(a1, JSFunction::kCodeOffset));
  __ RecordComment("]");

  __ RecordComment("[  JumpCodeObject(a2);");
  __ JumpCodeObject(a2);
  __ RecordComment("]");

  __ RecordComment("[  bind(&stack_overflow);");
  __ bind(&stack_overflow);
  __ RecordComment("]");
  {
    FrameScope frame(masm, StackFrame::MANUAL);
    __ RecordComment("[  CallRuntime(Runtime::kThrowStackOverflow);");
    __ CallRuntime(Runtime::kThrowStackOverflow);
    __ RecordComment("]");
    __ RecordComment("[  break_(0xCC);");
    __ break_(0xCC);
    __ RecordComment("]");
  }
}

void Builtins::Generate_WasmCompileLazy(MacroAssembler* masm) {
  // The function index was put in t0 by the jump table trampoline.
  // Convert to Smi for the runtime call
  __ RecordComment("[  SmiTag(kWasmCompileLazyFuncIndexRegister);");
  __ SmiTag(kWasmCompileLazyFuncIndexRegister);
  __ RecordComment("]");
  {
    HardAbortScope hard_abort(masm);  // Avoid calls to Abort.
    FrameScope scope(masm, StackFrame::WASM_COMPILE_LAZY);

    // Save all parameter registers (see kGpParamRegisters in wasm-linkage.cc).
    // They might be overwritten in the runtime call below. We don't have any
    // callee-saved registers in wasm, so no need to store anything else.
    constexpr RegList gp_regs = Register::ListOf(a0, a2, a3, a4, a5, a6, a7);
    constexpr RegList fp_regs =
        DoubleRegister::ListOf(fa0, fa1, fa2, fa3, fa4, fa5, fa6);
    static_assert(WasmCompileLazyFrameConstants::kNumberOfSavedGpParamRegs ==
                      arraysize(wasm::kGpParamRegisters),
                  "frame size mismatch");
    static_assert(WasmCompileLazyFrameConstants::kNumberOfSavedFpParamRegs ==
                      arraysize(wasm::kFpParamRegisters),
                  "frame size mismatch");
    __ RecordComment("[  MultiPush(gp_regs);");
    __ MultiPush(gp_regs);
    __ RecordComment("]");
    __ RecordComment("[  MultiPushFPU(fp_regs);");
    __ MultiPushFPU(fp_regs);
    __ RecordComment("]");

    // Pass instance and function index as an explicit arguments to the runtime
    // function.
    __ RecordComment(
        "[  Push(kWasmInstanceRegister, kWasmCompileLazyFuncIndexRegister);");
    __ Push(kWasmInstanceRegister, kWasmCompileLazyFuncIndexRegister);
    __ RecordComment("]");
    // Initialize the JavaScript context with 0. CEntry will use it to
    // set the current context on the isolate.
    __ RecordComment("[  Move(kContextRegister, Smi::zero());");
    __ Move(kContextRegister, Smi::zero());
    __ RecordComment("]");
    __ RecordComment("[  CallRuntime(Runtime::kWasmCompileLazy, 2);");
    __ CallRuntime(Runtime::kWasmCompileLazy, 2);
    __ RecordComment("]");

    __ RecordComment(
        "[  Move(s1, a0);  // move return value to s1 since a0 will be "
        "restored to");
    __ Move(s1, a0);  // move return value to s1 since a0 will be restored to
    __ RecordComment("]");
    // the value before the call

    // Restore registers.
    __ RecordComment("[  MultiPopFPU(fp_regs);");
    __ MultiPopFPU(fp_regs);
    __ RecordComment("]");
    __ RecordComment("[  MultiPop(gp_regs);");
    __ MultiPop(gp_regs);
    __ RecordComment("]");
  }
  // Finally, jump to the entrypoint.
  __ RecordComment("[  Jump(s1);");
  __ Jump(s1);
  __ RecordComment("]");
}

void Builtins::Generate_WasmDebugBreak(MacroAssembler* masm) {
  HardAbortScope hard_abort(masm);  // Avoid calls to Abort.
  {
    FrameScope scope(masm, StackFrame::WASM_DEBUG_BREAK);

    // Save all parameter registers. They might hold live values, we restore
    // them after the runtime call.
    __ RecordComment(
        "[  MultiPush(WasmDebugBreakFrameConstants::kPushedGpRegs);");
    __ MultiPush(WasmDebugBreakFrameConstants::kPushedGpRegs);
    __ RecordComment("]");
    __ RecordComment(
        "[  MultiPushFPU(WasmDebugBreakFrameConstants::kPushedFpRegs);");
    __ MultiPushFPU(WasmDebugBreakFrameConstants::kPushedFpRegs);
    __ RecordComment("]");

    // Initialize the JavaScript context with 0. CEntry will use it to
    // set the current context on the isolate.
    __ RecordComment("[  Move(cp, Smi::zero());");
    __ Move(cp, Smi::zero());
    __ RecordComment("]");
    __ RecordComment("[  CallRuntime(Runtime::kWasmDebugBreak, 0);");
    __ CallRuntime(Runtime::kWasmDebugBreak, 0);
    __ RecordComment("]");

    // Restore registers.
    __ RecordComment(
        "[  MultiPopFPU(WasmDebugBreakFrameConstants::kPushedFpRegs);");
    __ MultiPopFPU(WasmDebugBreakFrameConstants::kPushedFpRegs);
    __ RecordComment("]");
    __ RecordComment(
        "[  MultiPop(WasmDebugBreakFrameConstants::kPushedGpRegs);");
    __ MultiPop(WasmDebugBreakFrameConstants::kPushedGpRegs);
    __ RecordComment("]");
  }
  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");
}

void Builtins::Generate_CEntry(MacroAssembler* masm, int result_size,
                               SaveFPRegsMode save_doubles, ArgvMode argv_mode,
                               bool builtin_exit_frame) {
  // Called from JavaScript; parameters are on stack as if calling JS function
  // a0: number of arguments including receiver
  // a1: pointer to builtin function
  // fp: frame pointer    (restored after C call)
  // sp: stack pointer    (restored as callee's sp after C call)
  // cp: current context  (C callee-saved)
  //
  // If argv_mode == kArgvInRegister:
  // a2: pointer to the first argument

  if (argv_mode == kArgvInRegister) {
    // Move argv into the correct register.
    __ RecordComment("[  Move(s1, a2);");
    __ Move(s1, a2);
    __ RecordComment("]");
  } else {
    // Compute the argv pointer in a callee-saved register.
    __ RecordComment(
        "[  CalcScaledAddress(s1, sp, a0, kSystemPointerSizeLog2);");
    __ CalcScaledAddress(s1, sp, a0, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Sub64(s1, s1, kSystemPointerSize);");
    __ Sub64(s1, s1, kSystemPointerSize);
    __ RecordComment("]");
  }

  // Enter the exit frame that transitions from JavaScript to C++.
  FrameScope scope(masm, StackFrame::MANUAL);
  __ EnterExitFrame(
      save_doubles == kSaveFPRegs, 0,
      builtin_exit_frame ? StackFrame::BUILTIN_EXIT : StackFrame::EXIT);

  // s3: number of arguments  including receiver (C callee-saved)
  // s1: pointer to first argument (C callee-saved)
  // s2: pointer to builtin function (C callee-saved)

  // Prepare arguments for C routine.
  // a0 = argc
  __ RecordComment("[  Move(s3, a0);");
  __ Move(s3, a0);
  __ RecordComment("]");
  __ RecordComment("[  Move(s2, a1);");
  __ Move(s2, a1);
  __ RecordComment("]");

  // We are calling compiled C/C++ code. a0 and a1 hold our two arguments. We
  // also need to reserve the 4 argument slots on the stack.

  __ RecordComment("[  AssertStackIsAligned();");
  __ AssertStackIsAligned();
  __ RecordComment("]");

  // a0 = argc, a1 = argv, a2 = isolate
  __ RecordComment(
      "[  li(a2, ExternalReference::isolate_address(masm->isolate()));");
  __ li(a2, ExternalReference::isolate_address(masm->isolate()));
  __ RecordComment("]");
  __ RecordComment("[  Move(a1, s1);");
  __ Move(a1, s1);
  __ RecordComment("]");

  __ RecordComment("[  StoreReturnAddressAndCall(s2);");
  __ StoreReturnAddressAndCall(s2);
  __ RecordComment("]");

  // Result returned in a0 or a1:a0 - do not destroy these registers!

  // Check result for exception sentinel.
  Label exception_returned;
  __ RecordComment("[  LoadRoot(a4, RootIndex::kException);");
  __ LoadRoot(a4, RootIndex::kException);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&exception_returned, eq, a4, Operand(a0));");
  __ Branch(&exception_returned, eq, a4, Operand(a0));
  __ RecordComment("]");

  // Check that there is no pending exception, otherwise we
  // should have returned the exception sentinel.
  if (FLAG_debug_code) {
    Label okay;
    ExternalReference pending_exception_address = ExternalReference::Create(
        IsolateAddressId::kPendingExceptionAddress, masm->isolate());
    __ RecordComment("[  li(a2, pending_exception_address);");
    __ li(a2, pending_exception_address);
    __ RecordComment("]");
    __ RecordComment("[  Ld(a2, MemOperand(a2));");
    __ Ld(a2, MemOperand(a2));
    __ RecordComment("]");
    __ RecordComment("[  LoadRoot(a4, RootIndex::kTheHoleValue);");
    __ LoadRoot(a4, RootIndex::kTheHoleValue);
    __ RecordComment("]");
    // Cannot use check here as it attempts to generate call into runtime.
    __ RecordComment("[  Branch(&okay, eq, a4, Operand(a2));");
    __ Branch(&okay, eq, a4, Operand(a2));
    __ RecordComment("]");
    __ RecordComment("[  stop();");
    __ stop();
    __ RecordComment("]");
    __ RecordComment("[  bind(&okay);");
    __ bind(&okay);
    __ RecordComment("]");
  }

  // Exit C frame and return.
  // a0:a1: result
  // sp: stack pointer
  // fp: frame pointer
  Register argc = argv_mode == kArgvInRegister
                      // We don't want to pop arguments so set argc to no_reg.
                      ? no_reg
                      // s3: still holds argc (callee-saved).
                      : s3;
  __ RecordComment(
      "[  LeaveExitFrame(save_doubles == kSaveFPRegs, argc, EMIT_RETURN);");
  __ LeaveExitFrame(save_doubles == kSaveFPRegs, argc, EMIT_RETURN);
  __ RecordComment("]");

  // Handling of exception.
  __ RecordComment("[  bind(&exception_returned);");
  __ bind(&exception_returned);
  __ RecordComment("]");

  ExternalReference pending_handler_context_address = ExternalReference::Create(
      IsolateAddressId::kPendingHandlerContextAddress, masm->isolate());
  ExternalReference pending_handler_entrypoint_address =
      ExternalReference::Create(
          IsolateAddressId::kPendingHandlerEntrypointAddress, masm->isolate());
  ExternalReference pending_handler_fp_address = ExternalReference::Create(
      IsolateAddressId::kPendingHandlerFPAddress, masm->isolate());
  ExternalReference pending_handler_sp_address = ExternalReference::Create(
      IsolateAddressId::kPendingHandlerSPAddress, masm->isolate());

  // Ask the runtime for help to determine the handler. This will set a0 to
  // contain the current pending exception, don't clobber it.
  ExternalReference find_handler =
      ExternalReference::Create(Runtime::kUnwindAndFindExceptionHandler);
  {
    FrameScope scope(masm, StackFrame::MANUAL);
    __ RecordComment("[  PrepareCallCFunction(3, 0, a0);");
    __ PrepareCallCFunction(3, 0, a0);
    __ RecordComment("]");
    __ RecordComment("[  Move(a0, zero_reg);");
    __ Move(a0, zero_reg);
    __ RecordComment("]");
    __ RecordComment("[  Move(a1, zero_reg);");
    __ Move(a1, zero_reg);
    __ RecordComment("]");
    __ RecordComment(
        "[  li(a2, ExternalReference::isolate_address(masm->isolate()));");
    __ li(a2, ExternalReference::isolate_address(masm->isolate()));
    __ RecordComment("]");
    __ RecordComment("[  CallCFunction(find_handler, 3);");
    __ CallCFunction(find_handler, 3);
    __ RecordComment("]");
  }

  // Retrieve the handler context, SP and FP.
  __ RecordComment("[  li(cp, pending_handler_context_address);");
  __ li(cp, pending_handler_context_address);
  __ RecordComment("]");
  __ RecordComment("[  Ld(cp, MemOperand(cp));");
  __ Ld(cp, MemOperand(cp));
  __ RecordComment("]");
  __ RecordComment("[  li(sp, pending_handler_sp_address);");
  __ li(sp, pending_handler_sp_address);
  __ RecordComment("]");
  __ RecordComment("[  Ld(sp, MemOperand(sp));");
  __ Ld(sp, MemOperand(sp));
  __ RecordComment("]");
  __ RecordComment("[  li(fp, pending_handler_fp_address);");
  __ li(fp, pending_handler_fp_address);
  __ RecordComment("]");
  __ RecordComment("[  Ld(fp, MemOperand(fp));");
  __ Ld(fp, MemOperand(fp));
  __ RecordComment("]");

  // If the handler is a JS frame, restore the context to the frame. Note that
  // the context will be set to (cp == 0) for non-JS frames.
  Label zero;
  __ RecordComment("[  Branch(&zero, eq, cp, Operand(zero_reg));");
  __ Branch(&zero, eq, cp, Operand(zero_reg));
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(cp, MemOperand(fp, StandardFrameConstants::kContextOffset));");
  __ Sd(cp, MemOperand(fp, StandardFrameConstants::kContextOffset));
  __ RecordComment("]");
  __ RecordComment("[  bind(&zero);");
  __ bind(&zero);
  __ RecordComment("]");

  // Reset the masking register. This is done independent of the underlying
  // feature flag {FLAG_untrusted_code_mitigations} to make the snapshot work
  // with both configurations. It is safe to always do this, because the
  // underlying register is caller-saved and can be arbitrarily clobbered.
  __ RecordComment("[  ResetSpeculationPoisonRegister();");
  __ ResetSpeculationPoisonRegister();
  __ RecordComment("]");

  // Compute the handler entry address and jump to it.
  __ RecordComment("[  li(t6, pending_handler_entrypoint_address);");
  __ li(t6, pending_handler_entrypoint_address);
  __ RecordComment("]");
  __ RecordComment("[  Ld(t6, MemOperand(t6));");
  __ Ld(t6, MemOperand(t6));
  __ RecordComment("]");
  __ RecordComment("[  Jump(t6);");
  __ Jump(t6);
  __ RecordComment("]");
}

void Builtins::Generate_DoubleToI(MacroAssembler* masm) {
  Label done;
  Register result_reg = t0;

  Register scratch = GetRegisterThatIsNotOneOf(result_reg);
  Register scratch2 = GetRegisterThatIsNotOneOf(result_reg, scratch);
  Register scratch3 = GetRegisterThatIsNotOneOf(result_reg, scratch, scratch2);
  DoubleRegister double_scratch = kScratchDoubleReg;

  // Account for saved regs.
  const int kArgumentOffset = 4 * kSystemPointerSize;

  __ RecordComment("[  Push(result_reg);");
  __ Push(result_reg);
  __ RecordComment("]");
  __ RecordComment("[  Push(scratch, scratch2, scratch3);");
  __ Push(scratch, scratch2, scratch3);
  __ RecordComment("]");

  // Load double input.
  __ RecordComment(
      "[  LoadDouble(double_scratch, MemOperand(sp, kArgumentOffset));");
  __ LoadDouble(double_scratch, MemOperand(sp, kArgumentOffset));
  __ RecordComment("]");

  // Try a conversion to a signed integer, if exception occurs, scratch is
  // set to 0
  __ RecordComment("[  Trunc_w_d(scratch3, double_scratch, scratch);");
  __ Trunc_w_d(scratch3, double_scratch, scratch);
  __ RecordComment("]");

  // If we had no exceptions then set result_reg and we are done.
  Label error;
  __ RecordComment("[  Branch(&error, eq, scratch, Operand(zero_reg));");
  __ Branch(&error, eq, scratch, Operand(zero_reg));
  __ RecordComment("]");
  __ RecordComment("[  Move(result_reg, scratch3);");
  __ Move(result_reg, scratch3);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&done);");
  __ Branch(&done);
  __ RecordComment("]");
  __ RecordComment("[  bind(&error);");
  __ bind(&error);
  __ RecordComment("]");

  // Load the double value and perform a manual truncation.
  Register input_high = scratch2;
  Register input_low = scratch3;

  __ RecordComment(
      "[  Lw(input_low, MemOperand(sp, kArgumentOffset + "
      "Register::kMantissaOffset));");
  __ Lw(input_low, MemOperand(sp, kArgumentOffset + Register::kMantissaOffset));
  __ RecordComment("]");
  __ Lw(input_high,
        MemOperand(sp, kArgumentOffset + Register::kExponentOffset));

  Label normal_exponent;
  // Extract the biased exponent in result.
  __ ExtractBits(result_reg, input_high, HeapNumber::kExponentShift,
                 HeapNumber::kExponentBits);

  // Check for Infinity and NaNs, which should return 0.
  __ RecordComment("[  Sub32(scratch, result_reg, HeapNumber::kExponentMask);");
  __ Sub32(scratch, result_reg, HeapNumber::kExponentMask);
  __ RecordComment("]");
  __ LoadZeroIfConditionZero(
      result_reg,
      scratch);  // result_reg = scratch == 0 ? 0 : result_reg
  __ RecordComment("[  Branch(&done, eq, scratch, Operand(zero_reg));");
  __ Branch(&done, eq, scratch, Operand(zero_reg));
  __ RecordComment("]");

  // Express exponent as delta to (number of mantissa bits + 31).
  __ Sub32(result_reg, result_reg,
           Operand(HeapNumber::kExponentBias + HeapNumber::kMantissaBits + 31));

  // If the delta is strictly positive, all bits would be shifted away,
  // which means that we can return 0.
  __ RecordComment(
      "[  Branch(&normal_exponent, le, result_reg, Operand(zero_reg));");
  __ Branch(&normal_exponent, le, result_reg, Operand(zero_reg));
  __ RecordComment("]");
  __ RecordComment("[  Move(result_reg, zero_reg);");
  __ Move(result_reg, zero_reg);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&done);");
  __ Branch(&done);
  __ RecordComment("]");

  __ RecordComment("[  bind(&normal_exponent);");
  __ bind(&normal_exponent);
  __ RecordComment("]");
  const int kShiftBase = HeapNumber::kNonMantissaBitsInTopWord - 1;
  // Calculate shift.
  __ Add32(scratch, result_reg,
           Operand(kShiftBase + HeapNumber::kMantissaBits));

  // Save the sign.
  Register sign = result_reg;
  result_reg = no_reg;
  __ RecordComment("[  And(sign, input_high, Operand(HeapNumber::kSignMask));");
  __ And(sign, input_high, Operand(HeapNumber::kSignMask));
  __ RecordComment("]");

  // We must specially handle shifts greater than 31.
  Label high_shift_needed, high_shift_done;
  __ RecordComment("[  Branch(&high_shift_needed, lt, scratch, Operand(32));");
  __ Branch(&high_shift_needed, lt, scratch, Operand(32));
  __ RecordComment("]");
  __ RecordComment("[  Move(input_high, zero_reg);");
  __ Move(input_high, zero_reg);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&high_shift_done);");
  __ Branch(&high_shift_done);
  __ RecordComment("]");
  __ RecordComment("[  bind(&high_shift_needed);");
  __ bind(&high_shift_needed);
  __ RecordComment("]");

  // Set the implicit 1 before the mantissa part in input_high.
  __ Or(input_high, input_high,
        Operand(1 << HeapNumber::kMantissaBitsInTopWord));
  // Shift the mantissa bits to the correct position.
  // We don't need to clear non-mantissa bits as they will be shifted away.
  // If they weren't, it would mean that the answer is in the 32bit range.
  __ RecordComment("[  Sll32(input_high, input_high, scratch);");
  __ Sll32(input_high, input_high, scratch);
  __ RecordComment("]");

  __ RecordComment("[  bind(&high_shift_done);");
  __ bind(&high_shift_done);
  __ RecordComment("]");

  // Replace the shifted bits with bits from the lower mantissa word.
  Label pos_shift, shift_done, sign_negative;
  __ RecordComment("[  li(kScratchReg, 32);");
  __ li(kScratchReg, 32);
  __ RecordComment("]");
  __ RecordComment("[  subw(scratch, kScratchReg, scratch);");
  __ subw(scratch, kScratchReg, scratch);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&pos_shift, ge, scratch, Operand(zero_reg));");
  __ Branch(&pos_shift, ge, scratch, Operand(zero_reg));
  __ RecordComment("]");

  // Negate scratch.
  __ RecordComment("[  Sub32(scratch, zero_reg, scratch);");
  __ Sub32(scratch, zero_reg, scratch);
  __ RecordComment("]");
  __ RecordComment("[  Sll32(input_low, input_low, scratch);");
  __ Sll32(input_low, input_low, scratch);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&shift_done);");
  __ Branch(&shift_done);
  __ RecordComment("]");

  __ RecordComment("[  bind(&pos_shift);");
  __ bind(&pos_shift);
  __ RecordComment("]");
  __ RecordComment("[  srlw(input_low, input_low, scratch);");
  __ srlw(input_low, input_low, scratch);
  __ RecordComment("]");

  __ RecordComment("[  bind(&shift_done);");
  __ bind(&shift_done);
  __ RecordComment("]");
  __ RecordComment("[  Or(input_high, input_high, Operand(input_low));");
  __ Or(input_high, input_high, Operand(input_low));
  __ RecordComment("]");
  // Restore sign if necessary.
  __ RecordComment("[  Move(scratch, sign);");
  __ Move(scratch, sign);
  __ RecordComment("]");
  result_reg = sign;
  sign = no_reg;
  __ RecordComment("[  Sub32(result_reg, zero_reg, input_high);");
  __ Sub32(result_reg, zero_reg, input_high);
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&sign_negative, ne, scratch, Operand(zero_reg));");
  __ Branch(&sign_negative, ne, scratch, Operand(zero_reg));
  __ RecordComment("]");
  __ RecordComment("[  Move(result_reg, input_high);");
  __ Move(result_reg, input_high);
  __ RecordComment("]");
  __ RecordComment("[  bind(&sign_negative);");
  __ bind(&sign_negative);
  __ RecordComment("]");

  __ RecordComment("[  bind(&done);");
  __ bind(&done);
  __ RecordComment("]");

  __ RecordComment("[  Sd(result_reg, MemOperand(sp, kArgumentOffset));");
  __ Sd(result_reg, MemOperand(sp, kArgumentOffset));
  __ RecordComment("]");
  __ RecordComment("[  Pop(scratch, scratch2, scratch3);");
  __ Pop(scratch, scratch2, scratch3);
  __ RecordComment("]");
  __ RecordComment("[  Pop(result_reg);");
  __ Pop(result_reg);
  __ RecordComment("]");
  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");
}

void Builtins::Generate_GenericJSToWasmWrapper(MacroAssembler* masm) {
  // TODO(v8:10701): Implement for this platform.
  __ RecordComment("[  Trap();");
  __ Trap();
  __ RecordComment("]");
}

namespace {

int AddressOffset(ExternalReference ref0, ExternalReference ref1) {
  int64_t offset = (ref0.address() - ref1.address());
  DCHECK(static_cast<int>(offset) == offset);
  return static_cast<int>(offset);
}

// Calls an API function.  Allocates HandleScope, extracts returned value
// from handle and propagates exceptions.  Restores context.  stack_space
// - space to be unwound on exit (includes the call JS arguments space and
// the additional space allocated for the fast call).
void CallApiFunctionAndReturn(MacroAssembler* masm, Register function_address,
                              ExternalReference thunk_ref, int stack_space,
                              MemOperand* stack_space_operand,
                              MemOperand return_value_operand) {
  Isolate* isolate = masm->isolate();
  ExternalReference next_address =
      ExternalReference::handle_scope_next_address(isolate);
  const int kNextOffset = 0;
  const int kLimitOffset = AddressOffset(
      ExternalReference::handle_scope_limit_address(isolate), next_address);
  const int kLevelOffset = AddressOffset(
      ExternalReference::handle_scope_level_address(isolate), next_address);

  DCHECK(function_address == a1 || function_address == a2);

  Label profiler_enabled, end_profiler_check;
  __ RecordComment(
      "[  li(t6, ExternalReference::is_profiling_address(isolate));");
  __ li(t6, ExternalReference::is_profiling_address(isolate));
  __ RecordComment("]");
  __ RecordComment("[  Lb(t6, MemOperand(t6, 0));");
  __ Lb(t6, MemOperand(t6, 0));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&profiler_enabled, ne, t6, Operand(zero_reg));");
  __ Branch(&profiler_enabled, ne, t6, Operand(zero_reg));
  __ RecordComment("]");
  __ RecordComment(
      "[  li(t6, ExternalReference::address_of_runtime_stats_flag());");
  __ li(t6, ExternalReference::address_of_runtime_stats_flag());
  __ RecordComment("]");
  __ RecordComment("[  Lw(t6, MemOperand(t6, 0));");
  __ Lw(t6, MemOperand(t6, 0));
  __ RecordComment("]");
  __ RecordComment("[  Branch(&profiler_enabled, ne, t6, Operand(zero_reg));");
  __ Branch(&profiler_enabled, ne, t6, Operand(zero_reg));
  __ RecordComment("]");
  {
    // Call the api function directly.
    __ RecordComment("[  Move(t6, function_address);");
    __ Move(t6, function_address);
    __ RecordComment("]");
    __ RecordComment("[  Branch(&end_profiler_check);");
    __ Branch(&end_profiler_check);
    __ RecordComment("]");
  }

  __ RecordComment("[  bind(&profiler_enabled);");
  __ bind(&profiler_enabled);
  __ RecordComment("]");
  {
    // Additional parameter is the address of the actual callback.
    __ RecordComment("[  li(t6, thunk_ref);");
    __ li(t6, thunk_ref);
    __ RecordComment("]");
  }
  __ RecordComment("[  bind(&end_profiler_check);");
  __ bind(&end_profiler_check);
  __ RecordComment("]");

  // Allocate HandleScope in callee-save registers.
  __ RecordComment("[  li(s5, next_address);");
  __ li(s5, next_address);
  __ RecordComment("]");
  __ RecordComment("[  Ld(s3, MemOperand(s5, kNextOffset));");
  __ Ld(s3, MemOperand(s5, kNextOffset));
  __ RecordComment("]");
  __ RecordComment("[  Ld(s1, MemOperand(s5, kLimitOffset));");
  __ Ld(s1, MemOperand(s5, kLimitOffset));
  __ RecordComment("]");
  __ RecordComment("[  Lw(s2, MemOperand(s5, kLevelOffset));");
  __ Lw(s2, MemOperand(s5, kLevelOffset));
  __ RecordComment("]");
  __ RecordComment("[  Add32(s2, s2, Operand(1));");
  __ Add32(s2, s2, Operand(1));
  __ RecordComment("]");
  __ RecordComment("[  Sw(s2, MemOperand(s5, kLevelOffset));");
  __ Sw(s2, MemOperand(s5, kLevelOffset));
  __ RecordComment("]");

  __ RecordComment("[  StoreReturnAddressAndCall(t6);");
  __ StoreReturnAddressAndCall(t6);
  __ RecordComment("]");

  Label promote_scheduled_exception;
  Label delete_allocated_handles;
  Label leave_exit_frame;
  Label return_value_loaded;

  // Load value from ReturnValue.
  __ RecordComment("[  Ld(a0, return_value_operand);");
  __ Ld(a0, return_value_operand);
  __ RecordComment("]");
  __ RecordComment("[  bind(&return_value_loaded);");
  __ bind(&return_value_loaded);
  __ RecordComment("]");

  // No more valid handles (the result handle was the last one). Restore
  // previous handle scope.
  __ RecordComment("[  Sd(s3, MemOperand(s5, kNextOffset));");
  __ Sd(s3, MemOperand(s5, kNextOffset));
  __ RecordComment("]");
  if (__ emit_debug_code()) {
    __ RecordComment("[  Lw(a1, MemOperand(s5, kLevelOffset));");
    __ Lw(a1, MemOperand(s5, kLevelOffset));
    __ RecordComment("]");
    __ Check(eq, AbortReason::kUnexpectedLevelAfterReturnFromApiCall, a1,
             Operand(s2));
  }
  __ RecordComment("[  Sub32(s2, s2, Operand(1));");
  __ Sub32(s2, s2, Operand(1));
  __ RecordComment("]");
  __ RecordComment("[  Sw(s2, MemOperand(s5, kLevelOffset));");
  __ Sw(s2, MemOperand(s5, kLevelOffset));
  __ RecordComment("]");
  __ RecordComment("[  Ld(kScratchReg, MemOperand(s5, kLimitOffset));");
  __ Ld(kScratchReg, MemOperand(s5, kLimitOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&delete_allocated_handles, ne, s1, Operand(kScratchReg));");
  __ Branch(&delete_allocated_handles, ne, s1, Operand(kScratchReg));
  __ RecordComment("]");

  // Leave the API exit frame.
  __ RecordComment("[  bind(&leave_exit_frame);");
  __ bind(&leave_exit_frame);
  __ RecordComment("]");

  if (stack_space_operand == nullptr) {
    DCHECK_NE(stack_space, 0);
    __ RecordComment("[  li(s3, Operand(stack_space));");
    __ li(s3, Operand(stack_space));
    __ RecordComment("]");
  } else {
    DCHECK_EQ(stack_space, 0);
    STATIC_ASSERT(kCArgSlotCount == 0);
    __ RecordComment("[  Ld(s3, *stack_space_operand);");
    __ Ld(s3, *stack_space_operand);
    __ RecordComment("]");
  }

  static constexpr bool kDontSaveDoubles = false;
  static constexpr bool kRegisterContainsSlotCount = false;
  __ LeaveExitFrame(kDontSaveDoubles, s3, NO_EMIT_RETURN,
                    kRegisterContainsSlotCount);

  // Check if the function scheduled an exception.
  __ RecordComment("[  LoadRoot(a4, RootIndex::kTheHoleValue);");
  __ LoadRoot(a4, RootIndex::kTheHoleValue);
  __ RecordComment("]");
  __ RecordComment(
      "[  li(kScratchReg, "
      "ExternalReference::scheduled_exception_address(isolate));");
  __ li(kScratchReg, ExternalReference::scheduled_exception_address(isolate));
  __ RecordComment("]");
  __ RecordComment("[  Ld(a5, MemOperand(kScratchReg));");
  __ Ld(a5, MemOperand(kScratchReg));
  __ RecordComment("]");
  __ RecordComment(
      "[  Branch(&promote_scheduled_exception, ne, a4, Operand(a5));");
  __ Branch(&promote_scheduled_exception, ne, a4, Operand(a5));
  __ RecordComment("]");

  __ RecordComment("[  Ret();");
  __ Ret();
  __ RecordComment("]");

  // Re-throw by promoting a scheduled exception.
  __ RecordComment("[  bind(&promote_scheduled_exception);");
  __ bind(&promote_scheduled_exception);
  __ RecordComment("]");
  __ RecordComment("[  TailCallRuntime(Runtime::kPromoteScheduledException);");
  __ TailCallRuntime(Runtime::kPromoteScheduledException);
  __ RecordComment("]");

  // HandleScope limit has changed. Delete allocated extensions.
  __ RecordComment("[  bind(&delete_allocated_handles);");
  __ bind(&delete_allocated_handles);
  __ RecordComment("]");
  __ RecordComment("[  Sd(s1, MemOperand(s5, kLimitOffset));");
  __ Sd(s1, MemOperand(s5, kLimitOffset));
  __ RecordComment("]");
  __ RecordComment("[  Move(s3, a0);");
  __ Move(s3, a0);
  __ RecordComment("]");
  __ RecordComment("[  PrepareCallCFunction(1, s1);");
  __ PrepareCallCFunction(1, s1);
  __ RecordComment("]");
  __ RecordComment("[  li(a0, ExternalReference::isolate_address(isolate));");
  __ li(a0, ExternalReference::isolate_address(isolate));
  __ RecordComment("]");
  __ RecordComment(
      "[  CallCFunction(ExternalReference::delete_handle_scope_extensions(), "
      "1);");
  __ CallCFunction(ExternalReference::delete_handle_scope_extensions(), 1);
  __ RecordComment("]");
  __ RecordComment("[  Move(a0, s3);");
  __ Move(a0, s3);
  __ RecordComment("]");
  __ RecordComment("[  Branch(&leave_exit_frame);");
  __ Branch(&leave_exit_frame);
  __ RecordComment("]");
}

}  // namespace

void Builtins::Generate_CallApiCallback(MacroAssembler* masm) {
  // ----------- S t a t e -------------
  //  -- cp                  : context
  //  -- a1                  : api function address
  //  -- a2                  : arguments count (not including the receiver)
  //  -- a3                  : call data
  //  -- a0                  : holder
  //  --
  //  -- sp[0]               : receiver
  //  -- sp[8]               : first argument
  //  -- ...
  //  -- sp[(argc) * 8]      : last argument
  // -----------------------------------

  Register api_function_address = a1;
  Register argc = a2;
  Register call_data = a3;
  Register holder = a0;
  Register scratch = t0;
  Register base = t1;  // For addressing MemOperands on the stack.

  DCHECK(!AreAliased(api_function_address, argc, call_data, holder, scratch,
                     base));

  using FCA = FunctionCallbackArguments;

  STATIC_ASSERT(FCA::kArgsLength == 6);
  STATIC_ASSERT(FCA::kNewTargetIndex == 5);
  STATIC_ASSERT(FCA::kDataIndex == 4);
  STATIC_ASSERT(FCA::kReturnValueOffset == 3);
  STATIC_ASSERT(FCA::kReturnValueDefaultValueIndex == 2);
  STATIC_ASSERT(FCA::kIsolateIndex == 1);
  STATIC_ASSERT(FCA::kHolderIndex == 0);

  // Set up FunctionCallbackInfo's implicit_args on the stack as follows:
  //
  // Target state:
  //   sp[0 * kSystemPointerSize]: kHolder
  //   sp[1 * kSystemPointerSize]: kIsolate
  //   sp[2 * kSystemPointerSize]: undefined (kReturnValueDefaultValue)
  //   sp[3 * kSystemPointerSize]: undefined (kReturnValue)
  //   sp[4 * kSystemPointerSize]: kData
  //   sp[5 * kSystemPointerSize]: undefined (kNewTarget)

  // Set up the base register for addressing through MemOperands. It will point
  // at the receiver (located at sp + argc * kSystemPointerSize).
  __ RecordComment(
      "[  CalcScaledAddress(base, sp, argc, kSystemPointerSizeLog2);");
  __ CalcScaledAddress(base, sp, argc, kSystemPointerSizeLog2);
  __ RecordComment("]");

  // Reserve space on the stack.
  __ RecordComment(
      "[  Sub64(sp, sp, Operand(FCA::kArgsLength * kSystemPointerSize));");
  __ Sub64(sp, sp, Operand(FCA::kArgsLength * kSystemPointerSize));
  __ RecordComment("]");

  // kHolder.
  __ RecordComment("[  Sd(holder, MemOperand(sp, 0 * kSystemPointerSize));");
  __ Sd(holder, MemOperand(sp, 0 * kSystemPointerSize));
  __ RecordComment("]");

  // kIsolate.
  __ RecordComment(
      "[  li(scratch, ExternalReference::isolate_address(masm->isolate()));");
  __ li(scratch, ExternalReference::isolate_address(masm->isolate()));
  __ RecordComment("]");
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 1 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 1 * kSystemPointerSize));
  __ RecordComment("]");

  // kReturnValueDefaultValue and kReturnValue.
  __ RecordComment("[  LoadRoot(scratch, RootIndex::kUndefinedValue);");
  __ LoadRoot(scratch, RootIndex::kUndefinedValue);
  __ RecordComment("]");
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 2 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 2 * kSystemPointerSize));
  __ RecordComment("]");
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 3 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 3 * kSystemPointerSize));
  __ RecordComment("]");

  // kData.
  __ RecordComment("[  Sd(call_data, MemOperand(sp, 4 * kSystemPointerSize));");
  __ Sd(call_data, MemOperand(sp, 4 * kSystemPointerSize));
  __ RecordComment("]");

  // kNewTarget.
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 5 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 5 * kSystemPointerSize));
  __ RecordComment("]");

  // Keep a pointer to kHolder (= implicit_args) in a scratch register.
  // We use it below to set up the FunctionCallbackInfo object.
  __ RecordComment("[  Move(scratch, sp);");
  __ Move(scratch, sp);
  __ RecordComment("]");

  // Allocate the v8::Arguments structure in the arguments' space since
  // it's not controlled by GC.
  static constexpr int kApiStackSpace = 4;
  static constexpr bool kDontSaveDoubles = false;
  FrameScope frame_scope(masm, StackFrame::MANUAL);
  __ RecordComment("[  EnterExitFrame(kDontSaveDoubles, kApiStackSpace);");
  __ EnterExitFrame(kDontSaveDoubles, kApiStackSpace);
  __ RecordComment("]");

  // EnterExitFrame may align the sp.

  // FunctionCallbackInfo::implicit_args_ (points at kHolder as set up above).
  // Arguments are after the return address (pushed by EnterExitFrame()).
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 1 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 1 * kSystemPointerSize));
  __ RecordComment("]");

  // FunctionCallbackInfo::values_ (points at the first varargs argument passed
  // on the stack).
  __ Add64(scratch, scratch,
           Operand((FCA::kArgsLength + 1) * kSystemPointerSize));
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 2 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 2 * kSystemPointerSize));
  __ RecordComment("]");

  // FunctionCallbackInfo::length_.
  // Stored as int field, 32-bit integers within struct on stack always left
  // justified by n64 ABI.
  __ RecordComment("[  Sw(argc, MemOperand(sp, 3 * kSystemPointerSize));");
  __ Sw(argc, MemOperand(sp, 3 * kSystemPointerSize));
  __ RecordComment("]");

  // We also store the number of bytes to drop from the stack after returning
  // from the API function here.
  // Note: Unlike on other architectures, this stores the number of slots to
  // drop, not the number of bytes.
  __ RecordComment(
      "[  Add64(scratch, argc, Operand(FCA::kArgsLength + 1 /* receiver */));");
  __ Add64(scratch, argc, Operand(FCA::kArgsLength + 1 /* receiver */));
  __ RecordComment("]");
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 4 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 4 * kSystemPointerSize));
  __ RecordComment("]");

  // v8::InvocationCallback's argument.
  DCHECK(!AreAliased(api_function_address, scratch, a0));
  __ RecordComment("[  Add64(a0, sp, Operand(1 * kSystemPointerSize));");
  __ Add64(a0, sp, Operand(1 * kSystemPointerSize));
  __ RecordComment("]");

  ExternalReference thunk_ref = ExternalReference::invoke_function_callback();

  // There are two stack slots above the arguments we constructed on the stack.
  // TODO(jgruber): Document what these arguments are.
  static constexpr int kStackSlotsAboveFCA = 2;
  MemOperand return_value_operand(
      fp, (kStackSlotsAboveFCA + FCA::kReturnValueOffset) * kSystemPointerSize);

  static constexpr int kUseStackSpaceOperand = 0;
  MemOperand stack_space_operand(sp, 4 * kSystemPointerSize);

  AllowExternalCallThatCantCauseGC scope(masm);
  CallApiFunctionAndReturn(masm, api_function_address, thunk_ref,
                           kUseStackSpaceOperand, &stack_space_operand,
                           return_value_operand);
}

void Builtins::Generate_CallApiGetter(MacroAssembler* masm) {
  // Build v8::PropertyCallbackInfo::args_ array on the stack and push property
  // name below the exit frame to make GC aware of them.
  STATIC_ASSERT(PropertyCallbackArguments::kShouldThrowOnErrorIndex == 0);
  STATIC_ASSERT(PropertyCallbackArguments::kHolderIndex == 1);
  STATIC_ASSERT(PropertyCallbackArguments::kIsolateIndex == 2);
  STATIC_ASSERT(PropertyCallbackArguments::kReturnValueDefaultValueIndex == 3);
  STATIC_ASSERT(PropertyCallbackArguments::kReturnValueOffset == 4);
  STATIC_ASSERT(PropertyCallbackArguments::kDataIndex == 5);
  STATIC_ASSERT(PropertyCallbackArguments::kThisIndex == 6);
  STATIC_ASSERT(PropertyCallbackArguments::kArgsLength == 7);

  Register receiver = ApiGetterDescriptor::ReceiverRegister();
  Register holder = ApiGetterDescriptor::HolderRegister();
  Register callback = ApiGetterDescriptor::CallbackRegister();
  Register scratch = a4;
  DCHECK(!AreAliased(receiver, holder, callback, scratch));

  Register api_function_address = a2;

  // Here and below +1 is for name() pushed after the args_ array.
  using PCA = PropertyCallbackArguments;
  __ RecordComment(
      "[  Sub64(sp, sp, (PCA::kArgsLength + 1) * kSystemPointerSize);");
  __ Sub64(sp, sp, (PCA::kArgsLength + 1) * kSystemPointerSize);
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(receiver, MemOperand(sp, (PCA::kThisIndex + 1) * "
      "kSystemPointerSize));");
  __ Sd(receiver, MemOperand(sp, (PCA::kThisIndex + 1) * kSystemPointerSize));
  __ RecordComment("]");
  __ RecordComment(
      "[  LoadAnyTaggedField(scratch, FieldMemOperand(callback, "
      "AccessorInfo::kDataOffset));");
  __ LoadAnyTaggedField(scratch,
                        FieldMemOperand(callback, AccessorInfo::kDataOffset));
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(scratch, MemOperand(sp, (PCA::kDataIndex + 1) * "
      "kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, (PCA::kDataIndex + 1) * kSystemPointerSize));
  __ RecordComment("]");
  __ RecordComment("[  LoadRoot(scratch, RootIndex::kUndefinedValue);");
  __ LoadRoot(scratch, RootIndex::kUndefinedValue);
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(scratch, MemOperand(sp, (PCA::kReturnValueOffset + 1) * "
      "kSystemPointerSize));");
  __ Sd(scratch,
        MemOperand(sp, (PCA::kReturnValueOffset + 1) * kSystemPointerSize));
  __ RecordComment("]");
  __ Sd(scratch, MemOperand(sp, (PCA::kReturnValueDefaultValueIndex + 1) *
                                    kSystemPointerSize));
  __ RecordComment(
      "[  li(scratch, ExternalReference::isolate_address(masm->isolate()));");
  __ li(scratch, ExternalReference::isolate_address(masm->isolate()));
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(scratch, MemOperand(sp, (PCA::kIsolateIndex + 1) * "
      "kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, (PCA::kIsolateIndex + 1) * kSystemPointerSize));
  __ RecordComment("]");
  __ RecordComment(
      "[  Sd(holder, MemOperand(sp, (PCA::kHolderIndex + 1) * "
      "kSystemPointerSize));");
  __ Sd(holder, MemOperand(sp, (PCA::kHolderIndex + 1) * kSystemPointerSize));
  __ RecordComment("]");
  // should_throw_on_error -> false
  DCHECK_EQ(0, Smi::zero().ptr());
  __ Sd(zero_reg, MemOperand(sp, (PCA::kShouldThrowOnErrorIndex + 1) *
                                     kSystemPointerSize));
  __ RecordComment(
      "[  LoadTaggedPointerField(scratch, FieldMemOperand(callback, "
      "AccessorInfo::kNameOffset));");
  __ LoadTaggedPointerField(
      scratch, FieldMemOperand(callback, AccessorInfo::kNameOffset));
  __ RecordComment("]");
  __ RecordComment("[  Sd(scratch, MemOperand(sp, 0 * kSystemPointerSize));");
  __ Sd(scratch, MemOperand(sp, 0 * kSystemPointerSize));
  __ RecordComment("]");

  // v8::PropertyCallbackInfo::args_ array and name handle.
  const int kStackUnwindSpace = PropertyCallbackArguments::kArgsLength + 1;

  // Load address of v8::PropertyAccessorInfo::args_ array and name handle.
  __ RecordComment(
      "[  Move(a0, sp);                              // a0 = Handle<Name>");
  __ Move(a0, sp);  // a0 = Handle<Name>
  __ RecordComment("]");
  __ RecordComment(
      "[  Add64(a1, a0, Operand(1 * kSystemPointerSize));  // a1 = "
      "v8::PCI::args_");
  __ Add64(a1, a0, Operand(1 * kSystemPointerSize));  // a1 = v8::PCI::args_
  __ RecordComment("]");

  const int kApiStackSpace = 1;
  FrameScope frame_scope(masm, StackFrame::MANUAL);
  __ RecordComment("[  EnterExitFrame(false, kApiStackSpace);");
  __ EnterExitFrame(false, kApiStackSpace);
  __ RecordComment("]");

  // Create v8::PropertyCallbackInfo object on the stack and initialize
  // it's args_ field.
  __ RecordComment("[  Sd(a1, MemOperand(sp, 1 * kSystemPointerSize));");
  __ Sd(a1, MemOperand(sp, 1 * kSystemPointerSize));
  __ RecordComment("]");
  __ RecordComment("[  Add64(a1, sp, Operand(1 * kSystemPointerSize));");
  __ Add64(a1, sp, Operand(1 * kSystemPointerSize));
  __ RecordComment("]");
  // a1 = v8::PropertyCallbackInfo&

  ExternalReference thunk_ref =
      ExternalReference::invoke_accessor_getter_callback();

  __ RecordComment(
      "[  LoadTaggedPointerField(scratch, FieldMemOperand(callback, "
      "AccessorInfo::kJsGetterOffset));");
  __ LoadTaggedPointerField(
      scratch, FieldMemOperand(callback, AccessorInfo::kJsGetterOffset));
  __ RecordComment("]");
  __ Ld(api_function_address,
        FieldMemOperand(scratch, Foreign::kForeignAddressOffset));

  // +3 is to skip prolog, return address and name handle.
  MemOperand return_value_operand(
      fp,
      (PropertyCallbackArguments::kReturnValueOffset + 3) * kSystemPointerSize);
  MemOperand* const kUseStackSpaceConstant = nullptr;
  CallApiFunctionAndReturn(masm, api_function_address, thunk_ref,
                           kStackUnwindSpace, kUseStackSpaceConstant,
                           return_value_operand);
}

void Builtins::Generate_DirectCEntry(MacroAssembler* masm) {
  // The sole purpose of DirectCEntry is for movable callers (e.g. any general
  // purpose Code object) to be able to call into C functions that may trigger
  // GC and thus move the caller.
  //
  // DirectCEntry places the return address on the stack (updated by the GC),
  // making the call GC safe. The irregexp backend relies on this.

  // Make place for arguments to fit C calling convention. Callers use
  // EnterExitFrame/LeaveExitFrame so they handle stack restoring and we don't
  // have to do that here. Any caller must drop kCArgsSlotsSize stack space
  // after the call.
  __ RecordComment("[  Add64(sp, sp, -kCArgsSlotsSize);");
  __ Add64(sp, sp, -kCArgsSlotsSize);
  __ RecordComment("]");

  __ RecordComment(
      "[  Sd(ra, MemOperand(sp, kCArgsSlotsSize));  // Store the return "
      "address.");
  __ Sd(ra, MemOperand(sp, kCArgsSlotsSize));  // Store the return address.
  __ RecordComment("]");
  __ RecordComment(
      "[  Call(t6);                                 // Call the C++ function.");
  __ Call(t6);  // Call the C++ function.
  __ RecordComment("]");
  __ RecordComment(
      "[  Ld(t6, MemOperand(sp, kCArgsSlotsSize));  // Return to calling "
      "code.");
  __ Ld(t6, MemOperand(sp, kCArgsSlotsSize));  // Return to calling code.
  __ RecordComment("]");

  if (FLAG_debug_code && FLAG_enable_slow_asserts) {
    // In case of an error the return address may point to a memory area
    // filled with kZapValue by the GC. Dereference the address and check for
    // this.
    __ RecordComment("[  Uld(a4, MemOperand(t6));");
    __ Uld(a4, MemOperand(t6));
    __ RecordComment("]");
    __ Assert(ne, AbortReason::kReceivedInvalidReturnAddress, a4,
              Operand(reinterpret_cast<uint64_t>(kZapValue)));
  }

  __ RecordComment("[  Jump(t6);");
  __ Jump(t6);
  __ RecordComment("]");
}

#undef __

}  // namespace internal
}  // namespace v8

#endif  // V8_TARGET_ARCH_RISCV64
