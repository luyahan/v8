// Copyright 2014 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/codegen/assembler-inl.h"
#include "src/codegen/callable.h"
#include "src/codegen/macro-assembler.h"
#include "src/codegen/optimized-compilation-info.h"
#include "src/codegen/riscv64/constants-riscv64.h"
#include "src/compiler/backend/code-generator-impl.h"
#include "src/compiler/backend/code-generator.h"
#include "src/compiler/backend/gap-resolver.h"
#include "src/compiler/node-matchers.h"
#include "src/compiler/osr.h"
#include "src/heap/memory-chunk.h"
#include "src/wasm/wasm-code-manager.h"

namespace v8 {
namespace internal {
namespace compiler {

#define __ tasm()->

// TODO(plind): consider renaming these macros.
#define TRACE_MSG(msg)                                                      \
  PrintF("code_gen: \'%s\' in function %s at line %d\n", msg, __FUNCTION__, \
         __LINE__)

#define TRACE_UNIMPL()                                            \
  PrintF("UNIMPLEMENTED code_generator_riscv64: %s at line %d\n", \
         __FUNCTION__, __LINE__)

// Adds RISC-V-specific methods to convert InstructionOperands.
class RiscvOperandConverter final : public InstructionOperandConverter {
 public:
  RiscvOperandConverter(CodeGenerator* gen, Instruction* instr)
      : InstructionOperandConverter(gen, instr) {}

  FloatRegister OutputSingleRegister(size_t index = 0) {
    return ToSingleRegister(instr_->OutputAt(index));
  }

  FloatRegister InputSingleRegister(size_t index) {
    return ToSingleRegister(instr_->InputAt(index));
  }

  FloatRegister ToSingleRegister(InstructionOperand* op) {
    // Single (Float) and Double register namespace is same on RISC-V,
    // both are typedefs of FPURegister.
    return ToDoubleRegister(op);
  }

  Register InputOrZeroRegister(size_t index) {
    if (instr_->InputAt(index)->IsImmediate()) {
      DCHECK_EQ(0, InputInt32(index));
      return zero_reg;
    }
    return InputRegister(index);
  }

  DoubleRegister InputOrZeroDoubleRegister(size_t index) {
    if (instr_->InputAt(index)->IsImmediate()) return kDoubleRegZero;

    return InputDoubleRegister(index);
  }

  DoubleRegister InputOrZeroSingleRegister(size_t index) {
    if (instr_->InputAt(index)->IsImmediate()) return kDoubleRegZero;

    return InputSingleRegister(index);
  }

  Operand InputImmediate(size_t index) {
    Constant constant = ToConstant(instr_->InputAt(index));
    switch (constant.type()) {
      case Constant::kInt32:
        return Operand(constant.ToInt32());
      case Constant::kInt64:
        return Operand(constant.ToInt64());
      case Constant::kFloat32:
        return Operand::EmbeddedNumber(constant.ToFloat32());
      case Constant::kFloat64:
        return Operand::EmbeddedNumber(constant.ToFloat64().value());
      case Constant::kExternalReference:
      case Constant::kCompressedHeapObject:
      case Constant::kHeapObject:
        // TODO(plind): Maybe we should handle ExtRef & HeapObj here?
        //    maybe not done on arm due to const pool ??
        break;
      case Constant::kDelayedStringConstant:
        return Operand::EmbeddedStringConstant(
            constant.ToDelayedStringConstant());
      case Constant::kRpoNumber:
        UNREACHABLE();  // TODO(titzer): RPO immediates
        break;
    }
    UNREACHABLE();
  }

  Operand InputOperand(size_t index) {
    InstructionOperand* op = instr_->InputAt(index);
    if (op->IsRegister()) {
      return Operand(ToRegister(op));
    }
    return InputImmediate(index);
  }

  MemOperand MemoryOperand(size_t* first_index) {
    const size_t index = *first_index;
    switch (AddressingModeField::decode(instr_->opcode())) {
      case kMode_None:
        break;
      case kMode_MRI:
        *first_index += 2;
        return MemOperand(InputRegister(index + 0), InputInt32(index + 1));
      case kMode_MRR:
        // TODO(plind): r6 address mode, to be implemented ...
        UNREACHABLE();
    }
    UNREACHABLE();
  }

  MemOperand MemoryOperand(size_t index = 0) { return MemoryOperand(&index); }

  MemOperand ToMemOperand(InstructionOperand* op) const {
    DCHECK_NOT_NULL(op);
    DCHECK(op->IsStackSlot() || op->IsFPStackSlot());
    return SlotToMemOperand(AllocatedOperand::cast(op)->index());
  }

  MemOperand SlotToMemOperand(int slot) const {
    FrameOffset offset = frame_access_state()->GetFrameOffset(slot);
    return MemOperand(offset.from_stack_pointer() ? sp : fp, offset.offset());
  }
};

static inline bool HasRegisterInput(Instruction* instr, size_t index) {
  return instr->InputAt(index)->IsRegister();
}

namespace {

class OutOfLineRecordWrite final : public OutOfLineCode {
 public:
  OutOfLineRecordWrite(CodeGenerator* gen, Register object, Register index,
                       Register value, Register scratch0, Register scratch1,
                       RecordWriteMode mode, StubCallMode stub_mode)
      : OutOfLineCode(gen),
        object_(object),
        index_(index),
        value_(value),
        scratch0_(scratch0),
        scratch1_(scratch1),
        mode_(mode),
        stub_mode_(stub_mode),
        must_save_lr_(!gen->frame_access_state()->has_frame()),
        zone_(gen->zone()) {}

  void Generate() final {
    if (mode_ > RecordWriteMode::kValueIsPointer) {
      __ RecordComment("[  JumpIfSmi(value_, exit());");
      __ JumpIfSmi(value_, exit());
      __ RecordComment("]");
    }
    if (COMPRESS_POINTERS_BOOL) {
      __ RecordComment("[  DecompressTaggedPointer(value_, value_);");
      __ DecompressTaggedPointer(value_, value_);
      __ RecordComment("]");
    }
    __ CheckPageFlag(value_, scratch0_,
                     MemoryChunk::kPointersToHereAreInterestingMask, eq,
                     exit());
    __ RecordComment("[  Add64(scratch1_, object_, index_);");
    __ Add64(scratch1_, object_, index_);
    __ RecordComment("]");
    RememberedSetAction const remembered_set_action =
        mode_ > RecordWriteMode::kValueIsMap ? EMIT_REMEMBERED_SET
                                             : OMIT_REMEMBERED_SET;
    SaveFPRegsMode const save_fp_mode =
        frame()->DidAllocateDoubleRegisters() ? kSaveFPRegs : kDontSaveFPRegs;
    if (must_save_lr_) {
      // We need to save and restore ra if the frame was elided.
      __ RecordComment("[  Push(ra);");
      __ Push(ra);
      __ RecordComment("]");
    }
    if (mode_ == RecordWriteMode::kValueIsEphemeronKey) {
      __ RecordComment(
          "[  CallEphemeronKeyBarrier(object_, scratch1_, save_fp_mode);");
      __ CallEphemeronKeyBarrier(object_, scratch1_, save_fp_mode);
      __ RecordComment("]");
    } else if (stub_mode_ == StubCallMode::kCallWasmRuntimeStub) {
      // A direct call to a wasm runtime stub defined in this module.
      // Just encode the stub index. This will be patched when the code
      // is added to the native module and copied into wasm code space.
      __ CallRecordWriteStub(object_, scratch1_, remembered_set_action,
                             save_fp_mode, wasm::WasmCode::kRecordWrite);
    } else {
      __ CallRecordWriteStub(object_, scratch1_, remembered_set_action,
                             save_fp_mode);
    }
    if (must_save_lr_) {
      __ RecordComment("[  Pop(ra);");
      __ Pop(ra);
      __ RecordComment("]");
    }
  }

 private:
  Register const object_;
  Register const index_;
  Register const value_;
  Register const scratch0_;
  Register const scratch1_;
  RecordWriteMode const mode_;
  StubCallMode const stub_mode_;
  bool must_save_lr_;
  Zone* zone_;
};

Condition FlagsConditionToConditionCmp(FlagsCondition condition) {
  switch (condition) {
    case kEqual:
      return eq;
    case kNotEqual:
      return ne;
    case kSignedLessThan:
      return lt;
    case kSignedGreaterThanOrEqual:
      return ge;
    case kSignedLessThanOrEqual:
      return le;
    case kSignedGreaterThan:
      return gt;
    case kUnsignedLessThan:
      return Uless;
    case kUnsignedGreaterThanOrEqual:
      return Ugreater_equal;
    case kUnsignedLessThanOrEqual:
      return Uless_equal;
    case kUnsignedGreaterThan:
      return Ugreater;
    case kUnorderedEqual:
    case kUnorderedNotEqual:
      break;
    default:
      break;
  }
  UNREACHABLE();
}

Condition FlagsConditionToConditionTst(FlagsCondition condition) {
  switch (condition) {
    case kNotEqual:
      return ne;
    case kEqual:
      return eq;
    default:
      break;
  }
  UNREACHABLE();
}

Condition FlagsConditionToConditionOvf(FlagsCondition condition) {
  switch (condition) {
    case kOverflow:
      return ne;
    case kNotOverflow:
      return eq;
    default:
      break;
  }
  UNREACHABLE();
}

FPUCondition FlagsConditionToConditionCmpFPU(bool* predicate,
                                             FlagsCondition condition) {
  switch (condition) {
    case kEqual:
      *predicate = true;
      return EQ;
    case kNotEqual:
      *predicate = false;
      return EQ;
    case kUnsignedLessThan:
      *predicate = true;
      return LT;
    case kUnsignedGreaterThanOrEqual:
      *predicate = false;
      return LT;
    case kUnsignedLessThanOrEqual:
      *predicate = true;
      return LE;
    case kUnsignedGreaterThan:
      *predicate = false;
      return LE;
    case kUnorderedEqual:
    case kUnorderedNotEqual:
      *predicate = true;
      break;
    default:
      *predicate = true;
      break;
  }
  UNREACHABLE();
}

void EmitWordLoadPoisoningIfNeeded(CodeGenerator* codegen,
                                   InstructionCode opcode, Instruction* instr,
                                   RiscvOperandConverter const& i) {
  const MemoryAccessMode access_mode =
      static_cast<MemoryAccessMode>(MiscField::decode(opcode));
  if (access_mode == kMemoryAccessPoisoned) {
    Register value = i.OutputRegister();
    codegen->tasm()->And(value, value, kSpeculationPoisonRegister);
  }
}

}  // namespace

#define ASSEMBLE_ATOMIC_LOAD_INTEGER(asm_instr)                              \
  do {                                                                       \
    __ RecordComment("[ asm_instr(i.OutputRegister(), i.MemoryOperand());"); \
    __ asm_instr(i.OutputRegister(), i.MemoryOperand());                     \
    __ RecordComment("]");                                                   \
    __ RecordComment("[ sync();");                                           \
    __ sync();                                                               \
    __ RecordComment("]");                                                   \
  } while (0)

#define ASSEMBLE_ATOMIC_STORE_INTEGER(asm_instr)                      \
  do {                                                                \
    __ RecordComment("[ sync();");                                    \
    __ sync();                                                        \
    __ RecordComment("]");                                            \
    __ RecordComment(                                                 \
        "[ asm_instr(i.InputOrZeroRegister(2), i.MemoryOperand());"); \
    __ asm_instr(i.InputOrZeroRegister(2), i.MemoryOperand());        \
    __ RecordComment("]");                                            \
    __ RecordComment("[ sync();");                                    \
    __ sync();                                                        \
    __ RecordComment("]");                                            \
  } while (0)

#define ASSEMBLE_ATOMIC_BINOP(load_linked, store_conditional, bin_instr)       \
  do {                                                                         \
    Label binop;                                                               \
    __ RecordComment(                                                          \
        "[ Add64(i.TempRegister(0), i.InputRegister(0), "                      \
        "i.InputRegister(1));");                                               \
    __ Add64(i.TempRegister(0), i.InputRegister(0), i.InputRegister(1));       \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ bind(&binop);");                                       \
    __ bind(&binop);                                                           \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ load_linked(i.OutputRegister(0), MemOperand(i.TempRegister(0), "    \
        "0));");                                                               \
    __ load_linked(i.OutputRegister(0), MemOperand(i.TempRegister(0), 0));     \
    __ RecordComment("]");                                                     \
    __ bin_instr(i.TempRegister(1), i.OutputRegister(0),                       \
                 Operand(i.InputRegister(2)));                                 \
    __ RecordComment(                                                          \
        "[ store_conditional(i.TempRegister(1), "                              \
        "MemOperand(i.TempRegister(0), 0));");                                 \
    __ store_conditional(i.TempRegister(1), MemOperand(i.TempRegister(0), 0)); \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ BranchShort(&binop, ne, i.TempRegister(1), Operand(zero_reg));");   \
    __ BranchShort(&binop, ne, i.TempRegister(1), Operand(zero_reg));          \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
  } while (0)

#define ASSEMBLE_ATOMIC_BINOP_EXT(load_linked, store_conditional, sign_extend, \
                                  size, bin_instr, representation)             \
  do {                                                                         \
    Label binop;                                                               \
    __ RecordComment(                                                          \
        "[ Add64(i.TempRegister(0), i.InputRegister(0), "                      \
        "i.InputRegister(1));");                                               \
    __ Add64(i.TempRegister(0), i.InputRegister(0), i.InputRegister(1));       \
    __ RecordComment("]");                                                     \
    if (representation == 32) {                                                \
      __ RecordComment("[ And(i.TempRegister(3), i.TempRegister(0), 0x3);");   \
      __ And(i.TempRegister(3), i.TempRegister(0), 0x3);                       \
      __ RecordComment("]");                                                   \
    } else {                                                                   \
      DCHECK_EQ(representation, 64);                                           \
      __ RecordComment("[ And(i.TempRegister(3), i.TempRegister(0), 0x7);");   \
      __ And(i.TempRegister(3), i.TempRegister(0), 0x7);                       \
      __ RecordComment("]");                                                   \
    }                                                                          \
    __ Sub64(i.TempRegister(0), i.TempRegister(0),                             \
             Operand(i.TempRegister(3)));                                      \
    __ RecordComment("[ Sll32(i.TempRegister(3), i.TempRegister(3), 3);");     \
    __ Sll32(i.TempRegister(3), i.TempRegister(3), 3);                         \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ bind(&binop);");                                       \
    __ bind(&binop);                                                           \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ load_linked(i.TempRegister(1), MemOperand(i.TempRegister(0), "      \
        "0));");                                                               \
    __ load_linked(i.TempRegister(1), MemOperand(i.TempRegister(0), 0));       \
    __ RecordComment("]");                                                     \
    __ ExtractBits(i.OutputRegister(0), i.TempRegister(1), i.TempRegister(3),  \
                   size, sign_extend);                                         \
    __ bin_instr(i.TempRegister(2), i.OutputRegister(0),                       \
                 Operand(i.InputRegister(2)));                                 \
    __ InsertBits(i.TempRegister(1), i.TempRegister(2), i.TempRegister(3),     \
                  size);                                                       \
    __ RecordComment(                                                          \
        "[ store_conditional(i.TempRegister(1), "                              \
        "MemOperand(i.TempRegister(0), 0));");                                 \
    __ store_conditional(i.TempRegister(1), MemOperand(i.TempRegister(0), 0)); \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ BranchShort(&binop, ne, i.TempRegister(1), Operand(zero_reg));");   \
    __ BranchShort(&binop, ne, i.TempRegister(1), Operand(zero_reg));          \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
  } while (0)

#define ASSEMBLE_ATOMIC_EXCHANGE_INTEGER(load_linked, store_conditional)       \
  do {                                                                         \
    Label exchange;                                                            \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ bind(&exchange);");                                    \
    __ bind(&exchange);                                                        \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ Add64(i.TempRegister(0), i.InputRegister(0), "                      \
        "i.InputRegister(1));");                                               \
    __ Add64(i.TempRegister(0), i.InputRegister(0), i.InputRegister(1));       \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ load_linked(i.OutputRegister(0), MemOperand(i.TempRegister(0), "    \
        "0));");                                                               \
    __ load_linked(i.OutputRegister(0), MemOperand(i.TempRegister(0), 0));     \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ Move(i.TempRegister(1), i.InputRegister(2));");        \
    __ Move(i.TempRegister(1), i.InputRegister(2));                            \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ store_conditional(i.TempRegister(1), "                              \
        "MemOperand(i.TempRegister(0), 0));");                                 \
    __ store_conditional(i.TempRegister(1), MemOperand(i.TempRegister(0), 0)); \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ BranchShort(&exchange, ne, i.TempRegister(1), "                     \
        "Operand(zero_reg));");                                                \
    __ BranchShort(&exchange, ne, i.TempRegister(1), Operand(zero_reg));       \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
  } while (0)

#define ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(                                  \
    load_linked, store_conditional, sign_extend, size, representation)         \
  do {                                                                         \
    Label exchange;                                                            \
    __ RecordComment(                                                          \
        "[ Add64(i.TempRegister(0), i.InputRegister(0), "                      \
        "i.InputRegister(1));");                                               \
    __ Add64(i.TempRegister(0), i.InputRegister(0), i.InputRegister(1));       \
    __ RecordComment("]");                                                     \
    if (representation == 32) {                                                \
      __ RecordComment("[ And(i.TempRegister(1), i.TempRegister(0), 0x3);");   \
      __ And(i.TempRegister(1), i.TempRegister(0), 0x3);                       \
      __ RecordComment("]");                                                   \
    } else {                                                                   \
      DCHECK_EQ(representation, 64);                                           \
      __ RecordComment("[ And(i.TempRegister(1), i.TempRegister(0), 0x7);");   \
      __ And(i.TempRegister(1), i.TempRegister(0), 0x7);                       \
      __ RecordComment("]");                                                   \
    }                                                                          \
    __ Sub64(i.TempRegister(0), i.TempRegister(0),                             \
             Operand(i.TempRegister(1)));                                      \
    __ RecordComment("[ Sll32(i.TempRegister(1), i.TempRegister(1), 3);");     \
    __ Sll32(i.TempRegister(1), i.TempRegister(1), 3);                         \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ bind(&exchange);");                                    \
    __ bind(&exchange);                                                        \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ load_linked(i.TempRegister(2), MemOperand(i.TempRegister(0), "      \
        "0));");                                                               \
    __ load_linked(i.TempRegister(2), MemOperand(i.TempRegister(0), 0));       \
    __ RecordComment("]");                                                     \
    __ ExtractBits(i.OutputRegister(0), i.TempRegister(2), i.TempRegister(1),  \
                   size, sign_extend);                                         \
    __ InsertBits(i.TempRegister(2), i.InputRegister(2), i.TempRegister(1),    \
                  size);                                                       \
    __ RecordComment(                                                          \
        "[ store_conditional(i.TempRegister(2), "                              \
        "MemOperand(i.TempRegister(0), 0));");                                 \
    __ store_conditional(i.TempRegister(2), MemOperand(i.TempRegister(0), 0)); \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ BranchShort(&exchange, ne, i.TempRegister(2), "                     \
        "Operand(zero_reg));");                                                \
    __ BranchShort(&exchange, ne, i.TempRegister(2), Operand(zero_reg));       \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
  } while (0)

#define ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER(load_linked,                  \
                                                 store_conditional)            \
  do {                                                                         \
    Label compareExchange;                                                     \
    Label exit;                                                                \
    __ RecordComment(                                                          \
        "[ Add64(i.TempRegister(0), i.InputRegister(0), "                      \
        "i.InputRegister(1));");                                               \
    __ Add64(i.TempRegister(0), i.InputRegister(0), i.InputRegister(1));       \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ bind(&compareExchange);");                             \
    __ bind(&compareExchange);                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ load_linked(i.OutputRegister(0), MemOperand(i.TempRegister(0), "    \
        "0));");                                                               \
    __ load_linked(i.OutputRegister(0), MemOperand(i.TempRegister(0), 0));     \
    __ RecordComment("]");                                                     \
    __ BranchShort(&exit, ne, i.InputRegister(2),                              \
                   Operand(i.OutputRegister(0)));                              \
    __ RecordComment("[ Move(i.TempRegister(2), i.InputRegister(3));");        \
    __ Move(i.TempRegister(2), i.InputRegister(3));                            \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ store_conditional(i.TempRegister(2), "                              \
        "MemOperand(i.TempRegister(0), 0));");                                 \
    __ store_conditional(i.TempRegister(2), MemOperand(i.TempRegister(0), 0)); \
    __ RecordComment("]");                                                     \
    __ BranchShort(&compareExchange, ne, i.TempRegister(2),                    \
                   Operand(zero_reg));                                         \
    __ RecordComment("[ bind(&exit);");                                        \
    __ bind(&exit);                                                            \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
  } while (0)

#define ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(                          \
    load_linked, store_conditional, sign_extend, size, representation)         \
  do {                                                                         \
    Label compareExchange;                                                     \
    Label exit;                                                                \
    __ RecordComment(                                                          \
        "[ Add64(i.TempRegister(0), i.InputRegister(0), "                      \
        "i.InputRegister(1));");                                               \
    __ Add64(i.TempRegister(0), i.InputRegister(0), i.InputRegister(1));       \
    __ RecordComment("]");                                                     \
    if (representation == 32) {                                                \
      __ RecordComment("[ And(i.TempRegister(1), i.TempRegister(0), 0x3);");   \
      __ And(i.TempRegister(1), i.TempRegister(0), 0x3);                       \
      __ RecordComment("]");                                                   \
    } else {                                                                   \
      DCHECK_EQ(representation, 64);                                           \
      __ RecordComment("[ And(i.TempRegister(1), i.TempRegister(0), 0x7);");   \
      __ And(i.TempRegister(1), i.TempRegister(0), 0x7);                       \
      __ RecordComment("]");                                                   \
    }                                                                          \
    __ Sub64(i.TempRegister(0), i.TempRegister(0),                             \
             Operand(i.TempRegister(1)));                                      \
    __ RecordComment("[ Sll32(i.TempRegister(1), i.TempRegister(1), 3);");     \
    __ Sll32(i.TempRegister(1), i.TempRegister(1), 3);                         \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ bind(&compareExchange);");                             \
    __ bind(&compareExchange);                                                 \
    __ RecordComment("]");                                                     \
    __ RecordComment(                                                          \
        "[ load_linked(i.TempRegister(2), MemOperand(i.TempRegister(0), "      \
        "0));");                                                               \
    __ load_linked(i.TempRegister(2), MemOperand(i.TempRegister(0), 0));       \
    __ RecordComment("]");                                                     \
    __ ExtractBits(i.OutputRegister(0), i.TempRegister(2), i.TempRegister(1),  \
                   size, sign_extend);                                         \
    __ ExtractBits(i.InputRegister(2), i.InputRegister(2), i.TempRegister(1),  \
                   size, sign_extend);                                         \
    __ BranchShort(&exit, ne, i.InputRegister(2),                              \
                   Operand(i.OutputRegister(0)));                              \
    __ InsertBits(i.TempRegister(2), i.InputRegister(3), i.TempRegister(1),    \
                  size);                                                       \
    __ RecordComment(                                                          \
        "[ store_conditional(i.TempRegister(2), "                              \
        "MemOperand(i.TempRegister(0), 0));");                                 \
    __ store_conditional(i.TempRegister(2), MemOperand(i.TempRegister(0), 0)); \
    __ RecordComment("]");                                                     \
    __ BranchShort(&compareExchange, ne, i.TempRegister(2),                    \
                   Operand(zero_reg));                                         \
    __ RecordComment("[ bind(&exit);");                                        \
    __ bind(&exit);                                                            \
    __ RecordComment("]");                                                     \
    __ RecordComment("[ sync();");                                             \
    __ sync();                                                                 \
    __ RecordComment("]");                                                     \
  } while (0)

#define ASSEMBLE_IEEE754_BINOP(name)                                          \
  do {                                                                        \
    FrameScope scope(tasm(), StackFrame::MANUAL);                             \
    __ RecordComment("[ PrepareCallCFunction(0, 2, kScratchReg);");           \
    __ PrepareCallCFunction(0, 2, kScratchReg);                               \
    __ RecordComment("]");                                                    \
    __ MovToFloatParameters(i.InputDoubleRegister(0),                         \
                            i.InputDoubleRegister(1));                        \
    __ RecordComment(                                                         \
        "[ CallCFunction(ExternalReference::ieee754_##name##_function(), 0, " \
        "2);");                                                               \
    __ CallCFunction(ExternalReference::ieee754_##name##_function(), 0, 2);   \
    __ RecordComment("]");                                                    \
    /* Move the result in the double result register. */                      \
    __ RecordComment("[ MovFromFloatResult(i.OutputDoubleRegister());");      \
    __ MovFromFloatResult(i.OutputDoubleRegister());                          \
    __ RecordComment("]");                                                    \
  } while (0)

#define ASSEMBLE_IEEE754_UNOP(name)                                           \
  do {                                                                        \
    FrameScope scope(tasm(), StackFrame::MANUAL);                             \
    __ RecordComment("[ PrepareCallCFunction(0, 1, kScratchReg);");           \
    __ PrepareCallCFunction(0, 1, kScratchReg);                               \
    __ RecordComment("]");                                                    \
    __ RecordComment("[ MovToFloatParameter(i.InputDoubleRegister(0));");     \
    __ MovToFloatParameter(i.InputDoubleRegister(0));                         \
    __ RecordComment("]");                                                    \
    __ RecordComment(                                                         \
        "[ CallCFunction(ExternalReference::ieee754_##name##_function(), 0, " \
        "1);");                                                               \
    __ CallCFunction(ExternalReference::ieee754_##name##_function(), 0, 1);   \
    __ RecordComment("]");                                                    \
    /* Move the result in the double result register. */                      \
    __ RecordComment("[ MovFromFloatResult(i.OutputDoubleRegister());");      \
    __ MovFromFloatResult(i.OutputDoubleRegister());                          \
    __ RecordComment("]");                                                    \
  } while (0)

#define ASSEMBLE_F64X2_ARITHMETIC_BINOP(op)                     \
  do {                                                          \
    __ op(i.OutputSimd128Register(), i.InputSimd128Register(0), \
          i.InputSimd128Register(1));                           \
  } while (0)

void CodeGenerator::AssembleDeconstructFrame() {
  __ RecordComment("[  Move(sp, fp);");
  __ Move(sp, fp);
  __ RecordComment("]");
  __ RecordComment("[  Pop(ra, fp);");
  __ Pop(ra, fp);
  __ RecordComment("]");
}

void CodeGenerator::AssemblePrepareTailCall() {
  if (frame_access_state()->has_frame()) {
    __ RecordComment(
        "[  Ld(ra, MemOperand(fp, StandardFrameConstants::kCallerPCOffset));");
    __ Ld(ra, MemOperand(fp, StandardFrameConstants::kCallerPCOffset));
    __ RecordComment("]");
    __ RecordComment(
        "[  Ld(fp, MemOperand(fp, StandardFrameConstants::kCallerFPOffset));");
    __ Ld(fp, MemOperand(fp, StandardFrameConstants::kCallerFPOffset));
    __ RecordComment("]");
  }
  frame_access_state()->SetFrameAccessToSP();
}

void CodeGenerator::AssemblePopArgumentsAdaptorFrame(Register args_reg,
                                                     Register scratch1,
                                                     Register scratch2,
                                                     Register scratch3) {
  DCHECK(!AreAliased(args_reg, scratch1, scratch2, scratch3));
  Label done;

  // Check if current frame is an arguments adaptor frame.
  __ RecordComment(
      "[  LoadTaggedPointerField(scratch3, MemOperand(fp, "
      "StandardFrameConstants::kContextOffset));");
  __ LoadTaggedPointerField(
      scratch3, MemOperand(fp, StandardFrameConstants::kContextOffset));
  __ RecordComment("]");
  __ Branch(&done, ne, scratch3,
            Operand(StackFrame::TypeToMarker(StackFrame::ARGUMENTS_ADAPTOR)));

  // Load arguments count from current arguments adaptor frame (note, it
  // does not include receiver).
  Register caller_args_count_reg = scratch1;
  __ Ld(caller_args_count_reg,
        MemOperand(fp, ArgumentsAdaptorFrameConstants::kLengthOffset));
  __ RecordComment("[  SmiUntag(caller_args_count_reg);");
  __ SmiUntag(caller_args_count_reg);
  __ RecordComment("]");

  __ RecordComment(
      "[  PrepareForTailCall(args_reg, caller_args_count_reg, scratch2, "
      "scratch3);");
  __ PrepareForTailCall(args_reg, caller_args_count_reg, scratch2, scratch3);
  __ RecordComment("]");
  __ RecordComment("[  bind(&done);");
  __ bind(&done);
  __ RecordComment("]");
}

namespace {

void AdjustStackPointerForTailCall(TurboAssembler* tasm,
                                   FrameAccessState* state,
                                   int new_slot_above_sp,
                                   bool allow_shrinkage = true) {
  int current_sp_offset = state->GetSPToFPSlotCount() +
                          StandardFrameConstants::kFixedSlotCountAboveFp;
  int stack_slot_delta = new_slot_above_sp - current_sp_offset;
  if (stack_slot_delta > 0) {
    tasm->Sub64(sp, sp, stack_slot_delta * kSystemPointerSize);
    state->IncreaseSPDelta(stack_slot_delta);
  } else if (allow_shrinkage && stack_slot_delta < 0) {
    tasm->Add64(sp, sp, -stack_slot_delta * kSystemPointerSize);
    state->IncreaseSPDelta(stack_slot_delta);
  }
}

}  // namespace

void CodeGenerator::AssembleTailCallBeforeGap(Instruction* instr,
                                              int first_unused_stack_slot) {
  AdjustStackPointerForTailCall(tasm(), frame_access_state(),
                                first_unused_stack_slot, false);
}

void CodeGenerator::AssembleTailCallAfterGap(Instruction* instr,
                                             int first_unused_stack_slot) {
  AdjustStackPointerForTailCall(tasm(), frame_access_state(),
                                first_unused_stack_slot);
}

// Check that {kJavaScriptCallCodeStartRegister} is correct.
void CodeGenerator::AssembleCodeStartRegisterCheck() {
  __ RecordComment("[  ComputeCodeStartAddress(kScratchReg);");
  __ ComputeCodeStartAddress(kScratchReg);
  __ RecordComment("]");
  __ Assert(eq, AbortReason::kWrongFunctionCodeStart,
            kJavaScriptCallCodeStartRegister, Operand(kScratchReg));
}

// Check if the code object is marked for deoptimization. If it is, then it
// jumps to the CompileLazyDeoptimizedCode builtin. In order to do this we need
// to:
//    1. read from memory the word that contains that bit, which can be found in
//       the flags in the referenced {CodeDataContainer} object;
//    2. test kMarkedForDeoptimizationBit in those flags; and
//    3. if it is not zero then it jumps to the builtin.
void CodeGenerator::BailoutIfDeoptimized() {
  int offset = Code::kCodeDataContainerOffset - Code::kHeaderSize;
  __ RecordComment(
      "[  LoadTaggedPointerField(kScratchReg, "
      "MemOperand(kJavaScriptCallCodeStartRegister, offset));");
  __ LoadTaggedPointerField(
      kScratchReg, MemOperand(kJavaScriptCallCodeStartRegister, offset));
  __ RecordComment("]");
  __ Lw(kScratchReg,
        FieldMemOperand(kScratchReg,
                        CodeDataContainer::kKindSpecificFlagsOffset));
  __ And(kScratchReg, kScratchReg,
         Operand(1 << Code::kMarkedForDeoptimizationBit));
  __ Jump(BUILTIN_CODE(isolate(), CompileLazyDeoptimizedCode),
          RelocInfo::CODE_TARGET, ne, kScratchReg, Operand(zero_reg));
}

void CodeGenerator::GenerateSpeculationPoisonFromCodeStartRegister() {
  // Calculate a mask which has all bits set in the normal case, but has all
  // bits cleared if we are speculatively executing the wrong PC.
  //    difference = (current - expected) | (expected - current)
  //    poison = ~(difference >> (kBitsPerSystemPointer - 1))
  __ RecordComment("[  ComputeCodeStartAddress(kScratchReg);");
  __ ComputeCodeStartAddress(kScratchReg);
  __ RecordComment("]");
  __ RecordComment("[  Move(kSpeculationPoisonRegister, kScratchReg);");
  __ Move(kSpeculationPoisonRegister, kScratchReg);
  __ RecordComment("]");
  __ Sub32(kSpeculationPoisonRegister, kSpeculationPoisonRegister,
           kJavaScriptCallCodeStartRegister);
  __ Sub32(kJavaScriptCallCodeStartRegister, kJavaScriptCallCodeStartRegister,
           kScratchReg);
  __ or_(kSpeculationPoisonRegister, kSpeculationPoisonRegister,
         kJavaScriptCallCodeStartRegister);
  __ Sra64(kSpeculationPoisonRegister, kSpeculationPoisonRegister,
           kBitsPerSystemPointer - 1);
  __ Nor(kSpeculationPoisonRegister, kSpeculationPoisonRegister,
         kSpeculationPoisonRegister);
}

void CodeGenerator::AssembleRegisterArgumentPoisoning() {
  __ RecordComment(
      "[  And(kJSFunctionRegister, kJSFunctionRegister, "
      "kSpeculationPoisonRegister);");
  __ And(kJSFunctionRegister, kJSFunctionRegister, kSpeculationPoisonRegister);
  __ RecordComment("]");
  __ RecordComment(
      "[  And(kContextRegister, kContextRegister, "
      "kSpeculationPoisonRegister);");
  __ And(kContextRegister, kContextRegister, kSpeculationPoisonRegister);
  __ RecordComment("]");
  __ RecordComment("[  And(sp, sp, kSpeculationPoisonRegister);");
  __ And(sp, sp, kSpeculationPoisonRegister);
  __ RecordComment("]");
}

// Assembles an instruction after register allocation, producing machine code.
CodeGenerator::CodeGenResult CodeGenerator::AssembleArchInstruction(
    Instruction* instr) {
  RiscvOperandConverter i(this, instr);
  InstructionCode opcode = instr->opcode();
  ArchOpcode arch_opcode = ArchOpcodeField::decode(opcode);
  switch (arch_opcode) {
    case kArchCallCodeObject: {
      if (instr->InputAt(0)->IsImmediate()) {
        __ RecordComment("[  Call(i.InputCode(0), RelocInfo::CODE_TARGET);");
        __ Call(i.InputCode(0), RelocInfo::CODE_TARGET);
        __ RecordComment("]");
      } else {
        Register reg = i.InputRegister(0);
        DCHECK_IMPLIES(
            HasCallDescriptorFlag(instr, CallDescriptor::kFixedTargetRegister),
            reg == kJavaScriptCallCodeStartRegister);
        __ RecordComment("[  CallCodeObject(reg);");
        __ CallCodeObject(reg);
        __ RecordComment("]");
      }
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchCallBuiltinPointer: {
      DCHECK(!instr->InputAt(0)->IsImmediate());
      Register builtin_index = i.InputRegister(0);
      __ RecordComment("[  CallBuiltinByIndex(builtin_index);");
      __ CallBuiltinByIndex(builtin_index);
      __ RecordComment("]");
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchCallWasmFunction: {
      // FIXME (RISCV): isnt this test deadcode?
      if (arch_opcode == kArchTailCallCodeObjectFromJSFunction) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         i.TempRegister(0), i.TempRegister(1),
                                         i.TempRegister(2));
      }
      if (instr->InputAt(0)->IsImmediate()) {
        Constant constant = i.ToConstant(instr->InputAt(0));
        Address wasm_code = static_cast<Address>(constant.ToInt64());
        __ RecordComment("[  Call(wasm_code, constant.rmode());");
        __ Call(wasm_code, constant.rmode());
        __ RecordComment("]");
      } else {
        __ RecordComment("[  Add64(kScratchReg, i.InputRegister(0), 0);");
        __ Add64(kScratchReg, i.InputRegister(0), 0);
        __ RecordComment("]");
        __ RecordComment("[  Call(kScratchReg);");
        __ Call(kScratchReg);
        __ RecordComment("]");
      }
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchTailCallCodeObjectFromJSFunction:
    case kArchTailCallCodeObject: {
      if (arch_opcode == kArchTailCallCodeObjectFromJSFunction) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         i.TempRegister(0), i.TempRegister(1),
                                         i.TempRegister(2));
      }
      if (instr->InputAt(0)->IsImmediate()) {
        __ RecordComment("[  Jump(i.InputCode(0), RelocInfo::CODE_TARGET);");
        __ Jump(i.InputCode(0), RelocInfo::CODE_TARGET);
        __ RecordComment("]");
      } else {
        Register reg = i.InputRegister(0);
        DCHECK_IMPLIES(
            HasCallDescriptorFlag(instr, CallDescriptor::kFixedTargetRegister),
            reg == kJavaScriptCallCodeStartRegister);
        __ RecordComment("[  JumpCodeObject(reg);");
        __ JumpCodeObject(reg);
        __ RecordComment("]");
      }
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchTailCallWasm: {
      if (instr->InputAt(0)->IsImmediate()) {
        Constant constant = i.ToConstant(instr->InputAt(0));
        Address wasm_code = static_cast<Address>(constant.ToInt64());
        __ RecordComment("[  Jump(wasm_code, constant.rmode());");
        __ Jump(wasm_code, constant.rmode());
        __ RecordComment("]");
      } else {
        __ RecordComment("[  Add64(kScratchReg, i.InputRegister(0), 0);");
        __ Add64(kScratchReg, i.InputRegister(0), 0);
        __ RecordComment("]");
        __ RecordComment("[  Jump(kScratchReg);");
        __ Jump(kScratchReg);
        __ RecordComment("]");
      }
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchTailCallAddress: {
      CHECK(!instr->InputAt(0)->IsImmediate());
      Register reg = i.InputRegister(0);
      DCHECK_IMPLIES(
          HasCallDescriptorFlag(instr, CallDescriptor::kFixedTargetRegister),
          reg == kJavaScriptCallCodeStartRegister);
      __ RecordComment("[  Jump(reg);");
      __ Jump(reg);
      __ RecordComment("]");
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchCallJSFunction: {
      Register func = i.InputRegister(0);
      if (FLAG_debug_code) {
        // Check the function's context matches the context argument.
        __ RecordComment(
            "[  LoadTaggedPointerField(kScratchReg, FieldMemOperand(func, "
            "JSFunction::kContextOffset));");
        __ LoadTaggedPointerField(
            kScratchReg, FieldMemOperand(func, JSFunction::kContextOffset));
        __ RecordComment("]");
        __ Assert(eq, AbortReason::kWrongFunctionContext, cp,
                  Operand(kScratchReg));
      }
      static_assert(kJavaScriptCallCodeStartRegister == a2, "ABI mismatch");
      __ RecordComment(
          "[  LoadTaggedPointerField(a2, FieldMemOperand(func, "
          "JSFunction::kCodeOffset));");
      __ LoadTaggedPointerField(a2,
                                FieldMemOperand(func, JSFunction::kCodeOffset));
      __ RecordComment("]");
      __ RecordComment("[  CallCodeObject(a2);");
      __ CallCodeObject(a2);
      __ RecordComment("]");
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchPrepareCallCFunction: {
      int const num_parameters = MiscField::decode(instr->opcode());
      __ RecordComment("[  PrepareCallCFunction(num_parameters, kScratchReg);");
      __ PrepareCallCFunction(num_parameters, kScratchReg);
      __ RecordComment("]");
      // Frame alignment requires using FP-relative frame addressing.
      frame_access_state()->SetFrameAccessToFP();
      break;
    }
    case kArchSaveCallerRegisters: {
      fp_mode_ =
          static_cast<SaveFPRegsMode>(MiscField::decode(instr->opcode()));
      DCHECK(fp_mode_ == kDontSaveFPRegs || fp_mode_ == kSaveFPRegs);
      // kReturnRegister0 should have been saved before entering the stub.
      __ RecordComment("[  PushCallerSaved(fp_mode_, kReturnRegister0);");
      int bytes = __ PushCallerSaved(fp_mode_, kReturnRegister0);
      __ RecordComment("]");
      DCHECK(IsAligned(bytes, kSystemPointerSize));
      DCHECK_EQ(0, frame_access_state()->sp_delta());
      frame_access_state()->IncreaseSPDelta(bytes / kSystemPointerSize);
      DCHECK(!caller_registers_saved_);
      caller_registers_saved_ = true;
      break;
    }
    case kArchRestoreCallerRegisters: {
      DCHECK(fp_mode_ ==
             static_cast<SaveFPRegsMode>(MiscField::decode(instr->opcode())));
      DCHECK(fp_mode_ == kDontSaveFPRegs || fp_mode_ == kSaveFPRegs);
      // Don't overwrite the returned value.
      __ RecordComment("[  PopCallerSaved(fp_mode_, kReturnRegister0);");
      int bytes = __ PopCallerSaved(fp_mode_, kReturnRegister0);
      __ RecordComment("]");
      frame_access_state()->IncreaseSPDelta(-(bytes / kSystemPointerSize));
      DCHECK_EQ(0, frame_access_state()->sp_delta());
      DCHECK(caller_registers_saved_);
      caller_registers_saved_ = false;
      break;
    }
    case kArchPrepareTailCall:
      AssemblePrepareTailCall();
      break;
    case kArchCallCFunction: {
      int const num_parameters = MiscField::decode(instr->opcode());
      Label after_call;
      bool isWasmCapiFunction =
          linkage()->GetIncomingDescriptor()->IsWasmCapiFunction();
      if (isWasmCapiFunction) {
        // Put the return address in a stack slot.
        __ RecordComment(
            "[  LoadAddress(kScratchReg, &after_call, "
            "RelocInfo::EXTERNAL_REFERENCE);");
        __ LoadAddress(kScratchReg, &after_call, RelocInfo::EXTERNAL_REFERENCE);
        __ RecordComment("]");
        __ Sd(kScratchReg,
              MemOperand(fp, WasmExitFrameConstants::kCallingPCOffset));
      }
      if (instr->InputAt(0)->IsImmediate()) {
        ExternalReference ref = i.InputExternalReference(0);
        __ RecordComment("[  CallCFunction(ref, num_parameters);");
        __ CallCFunction(ref, num_parameters);
        __ RecordComment("]");
      } else {
        Register func = i.InputRegister(0);
        __ RecordComment("[  CallCFunction(func, num_parameters);");
        __ CallCFunction(func, num_parameters);
        __ RecordComment("]");
      }
      __ RecordComment("[  bind(&after_call);");
      __ bind(&after_call);
      __ RecordComment("]");
      if (isWasmCapiFunction) {
        RecordSafepoint(instr->reference_map(), Safepoint::kNoLazyDeopt);
      }

      frame_access_state()->SetFrameAccessToDefault();
      // Ideally, we should decrement SP delta to match the change of stack
      // pointer in CallCFunction. However, for certain architectures (e.g.
      // ARM), there may be more strict alignment requirement, causing old SP
      // to be saved on the stack. In those cases, we can not calculate the SP
      // delta statically.
      frame_access_state()->ClearSPDelta();
      if (caller_registers_saved_) {
        // Need to re-sync SP delta introduced in kArchSaveCallerRegisters.
        // Here, we assume the sequence to be:
        //   kArchSaveCallerRegisters;
        //   kArchCallCFunction;
        //   kArchRestoreCallerRegisters;
        int bytes =
            __ RequiredStackSizeForCallerSaved(fp_mode_, kReturnRegister0);
        frame_access_state()->IncreaseSPDelta(bytes / kSystemPointerSize);
      }
      break;
    }
    case kArchJmp:
      AssembleArchJump(i.InputRpo(0));
      break;
    case kArchBinarySearchSwitch:
      AssembleArchBinarySearchSwitch(instr);
      break;
    case kArchTableSwitch:
      AssembleArchTableSwitch(instr);
      break;
    case kArchAbortCSAAssert:
      DCHECK(i.InputRegister(0) == a0);
      {
        // We don't actually want to generate a pile of code for this, so just
        // claim there is a stack frame, without generating one.
        FrameScope scope(tasm(), StackFrame::NONE);
        __ Call(
            isolate()->builtins()->builtin_handle(Builtins::kAbortCSAAssert),
            RelocInfo::CODE_TARGET);
      }
      __ RecordComment("[  stop();");
      __ stop();
      __ RecordComment("]");
      break;
    case kArchDebugBreak:
      __ RecordComment("[  DebugBreak();");
      __ DebugBreak();
      __ RecordComment("]");
      break;
    case kArchComment:
      __ RecordComment(reinterpret_cast<const char*>(i.InputInt64(0)));
      break;
    case kArchNop:
    case kArchThrowTerminator:
      // don't emit code for nops.
      break;
    case kArchDeoptimize: {
      DeoptimizationExit* exit =
          BuildTranslation(instr, -1, 0, OutputFrameStateCombine::Ignore());
      CodeGenResult result = AssembleDeoptimizerCall(exit);
      if (result != kSuccess) return result;
      break;
    }
    case kArchRet:
      AssembleReturn(instr->InputAt(0));
      break;
    case kArchStackPointerGreaterThan:
      // Pseudo-instruction used for cmp/branch. No opcode emitted here.
      break;
    case kArchStackCheckOffset:
      __ RecordComment(
          "[  Move(i.OutputRegister(), Smi::FromInt(GetStackCheckOffset()));");
      __ Move(i.OutputRegister(), Smi::FromInt(GetStackCheckOffset()));
      __ RecordComment("]");
      break;
    case kArchFramePointer:
      __ RecordComment("[  Move(i.OutputRegister(), fp);");
      __ Move(i.OutputRegister(), fp);
      __ RecordComment("]");
      break;
    case kArchParentFramePointer:
      if (frame_access_state()->has_frame()) {
        __ RecordComment("[  Ld(i.OutputRegister(), MemOperand(fp, 0));");
        __ Ld(i.OutputRegister(), MemOperand(fp, 0));
        __ RecordComment("]");
      } else {
        __ RecordComment("[  Move(i.OutputRegister(), fp);");
        __ Move(i.OutputRegister(), fp);
        __ RecordComment("]");
      }
      break;
    case kArchTruncateDoubleToI:
      __ TruncateDoubleToI(isolate(), zone(), i.OutputRegister(),
                           i.InputDoubleRegister(0), DetermineStubCallMode());
      break;
    case kArchStoreWithWriteBarrier: {
      RecordWriteMode mode =
          static_cast<RecordWriteMode>(MiscField::decode(instr->opcode()));
      Register object = i.InputRegister(0);
      Register index = i.InputRegister(1);
      Register value = i.InputRegister(2);
      Register scratch0 = i.TempRegister(0);
      Register scratch1 = i.TempRegister(1);
      auto ool = zone()->New<OutOfLineRecordWrite>(this, object, index, value,
                                                   scratch0, scratch1, mode,
                                                   DetermineStubCallMode());
      __ RecordComment("[  Add64(kScratchReg, object, index);");
      __ Add64(kScratchReg, object, index);
      __ RecordComment("]");
      __ RecordComment("[  StoreTaggedField(value, MemOperand(kScratchReg));");
      __ StoreTaggedField(value, MemOperand(kScratchReg));
      __ RecordComment("]");
      __ RecordComment("__ CheckPageFlag");
      __ CheckPageFlag(object, scratch0,
                       MemoryChunk::kPointersFromHereAreInterestingMask, ne,
                       ool->entry());
      __ RecordComment("]");
      __ RecordComment("[  bind(ool->exit());");
      __ bind(ool->exit());
      __ RecordComment("]");
      break;
    }
    case kArchStackSlot: {
      FrameOffset offset =
          frame_access_state()->GetFrameOffset(i.InputInt32(0));
      Register base_reg = offset.from_stack_pointer() ? sp : fp;
      __ RecordComment(
          "[  Add64(i.OutputRegister(), base_reg, Operand(offset.offset()));");
      __ Add64(i.OutputRegister(), base_reg, Operand(offset.offset()));
      __ RecordComment("]");
      int alignment = i.InputInt32(1);
      DCHECK(alignment == 0 || alignment == 4 || alignment == 8 ||
             alignment == 16);
      if (FLAG_debug_code && alignment > 0) {
        // Verify that the output_register is properly aligned
        __ And(kScratchReg, i.OutputRegister(),
               Operand(kSystemPointerSize - 1));
        __ Assert(eq, AbortReason::kAllocationIsNotDoubleAligned, kScratchReg,
                  Operand(zero_reg));
      }
      if (alignment == 2 * kSystemPointerSize) {
        Label done;
        __ RecordComment(
            "[  Add64(kScratchReg, base_reg, Operand(offset.offset()));");
        __ Add64(kScratchReg, base_reg, Operand(offset.offset()));
        __ RecordComment("]");
        __ RecordComment(
            "[  And(kScratchReg, kScratchReg, Operand(alignment - 1));");
        __ And(kScratchReg, kScratchReg, Operand(alignment - 1));
        __ RecordComment("]");
        __ RecordComment(
            "[  BranchShort(&done, eq, kScratchReg, Operand(zero_reg));");
        __ BranchShort(&done, eq, kScratchReg, Operand(zero_reg));
        __ RecordComment("]");
        __ RecordComment(
            "[  Add64(i.OutputRegister(), i.OutputRegister(), "
            "kSystemPointerSize);");
        __ Add64(i.OutputRegister(), i.OutputRegister(), kSystemPointerSize);
        __ RecordComment("]");
        __ RecordComment("[  bind(&done);");
        __ bind(&done);
        __ RecordComment("]");
      } else if (alignment > 2 * kSystemPointerSize) {
        Label done;
        __ RecordComment(
            "[  Add64(kScratchReg, base_reg, Operand(offset.offset()));");
        __ Add64(kScratchReg, base_reg, Operand(offset.offset()));
        __ RecordComment("]");
        __ RecordComment(
            "[  And(kScratchReg, kScratchReg, Operand(alignment - 1));");
        __ And(kScratchReg, kScratchReg, Operand(alignment - 1));
        __ RecordComment("]");
        __ RecordComment(
            "[  BranchShort(&done, eq, kScratchReg, Operand(zero_reg));");
        __ BranchShort(&done, eq, kScratchReg, Operand(zero_reg));
        __ RecordComment("]");
        __ RecordComment("[  li(kScratchReg2, alignment);");
        __ li(kScratchReg2, alignment);
        __ RecordComment("]");
        __ RecordComment(
            "[  Sub64(kScratchReg2, kScratchReg2, Operand(kScratchReg));");
        __ Sub64(kScratchReg2, kScratchReg2, Operand(kScratchReg));
        __ RecordComment("]");
        __ RecordComment(
            "[  Add64(i.OutputRegister(), i.OutputRegister(), kScratchReg2);");
        __ Add64(i.OutputRegister(), i.OutputRegister(), kScratchReg2);
        __ RecordComment("]");
        __ RecordComment("[  bind(&done);");
        __ bind(&done);
        __ RecordComment("]");
      }

      break;
    }
    case kArchWordPoisonOnSpeculation:
      __ And(i.OutputRegister(), i.InputRegister(0),
             kSpeculationPoisonRegister);
      break;
    case kIeee754Float64Acos:
      ASSEMBLE_IEEE754_UNOP(acos);
      break;
    case kIeee754Float64Acosh:
      ASSEMBLE_IEEE754_UNOP(acosh);
      break;
    case kIeee754Float64Asin:
      ASSEMBLE_IEEE754_UNOP(asin);
      break;
    case kIeee754Float64Asinh:
      ASSEMBLE_IEEE754_UNOP(asinh);
      break;
    case kIeee754Float64Atan:
      ASSEMBLE_IEEE754_UNOP(atan);
      break;
    case kIeee754Float64Atanh:
      ASSEMBLE_IEEE754_UNOP(atanh);
      break;
    case kIeee754Float64Atan2:
      ASSEMBLE_IEEE754_BINOP(atan2);
      break;
    case kIeee754Float64Cos:
      ASSEMBLE_IEEE754_UNOP(cos);
      break;
    case kIeee754Float64Cosh:
      ASSEMBLE_IEEE754_UNOP(cosh);
      break;
    case kIeee754Float64Cbrt:
      ASSEMBLE_IEEE754_UNOP(cbrt);
      break;
    case kIeee754Float64Exp:
      ASSEMBLE_IEEE754_UNOP(exp);
      break;
    case kIeee754Float64Expm1:
      ASSEMBLE_IEEE754_UNOP(expm1);
      break;
    case kIeee754Float64Log:
      ASSEMBLE_IEEE754_UNOP(log);
      break;
    case kIeee754Float64Log1p:
      ASSEMBLE_IEEE754_UNOP(log1p);
      break;
    case kIeee754Float64Log2:
      ASSEMBLE_IEEE754_UNOP(log2);
      break;
    case kIeee754Float64Log10:
      ASSEMBLE_IEEE754_UNOP(log10);
      break;
    case kIeee754Float64Pow:
      ASSEMBLE_IEEE754_BINOP(pow);
      break;
    case kIeee754Float64Sin:
      ASSEMBLE_IEEE754_UNOP(sin);
      break;
    case kIeee754Float64Sinh:
      ASSEMBLE_IEEE754_UNOP(sinh);
      break;
    case kIeee754Float64Tan:
      ASSEMBLE_IEEE754_UNOP(tan);
      break;
    case kIeee754Float64Tanh:
      ASSEMBLE_IEEE754_UNOP(tanh);
      break;
    case kRiscvAdd32:
      __ RecordComment(
          "[  Add32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Add32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvAdd64:
      __ RecordComment(
          "[  Add64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Add64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvAddOvf64:
      __ AddOverflow64(i.OutputRegister(), i.InputRegister(0),
                       i.InputOperand(1), kScratchReg);
      break;
    case kRiscvSub32:
      __ RecordComment(
          "[  Sub32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Sub32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvSub64:
      __ RecordComment(
          "[  Sub64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Sub64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvSubOvf64:
      __ SubOverflow64(i.OutputRegister(), i.InputRegister(0),
                       i.InputOperand(1), kScratchReg);
      break;
    case kRiscvMul32:
      __ RecordComment(
          "[  Mul32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Mul32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvMulOvf32:
      __ MulOverflow32(i.OutputRegister(), i.InputRegister(0),
                       i.InputOperand(1), kScratchReg);
      break;
    case kRiscvMulHigh32:
      __ RecordComment(
          "[  Mulh32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Mulh32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvMulHighU32:
      __ Mulhu32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1),
                 kScratchReg, kScratchReg2);
      break;
    case kRiscvMulHigh64:
      __ RecordComment(
          "[  Mulh64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Mulh64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvDiv32: {
      __ RecordComment(
          "[  Div32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Div32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      // Set ouput to zero if divisor == 0
      __ RecordComment(
          "[  LoadZeroIfConditionZero(i.OutputRegister(), "
          "i.InputRegister(1));");
      __ LoadZeroIfConditionZero(i.OutputRegister(), i.InputRegister(1));
      __ RecordComment("]");
      break;
    }
    case kRiscvDivU32: {
      __ RecordComment(
          "[  Divu32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Divu32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      // Set ouput to zero if divisor == 0
      __ RecordComment(
          "[  LoadZeroIfConditionZero(i.OutputRegister(), "
          "i.InputRegister(1));");
      __ LoadZeroIfConditionZero(i.OutputRegister(), i.InputRegister(1));
      __ RecordComment("]");
      break;
    }
    case kRiscvMod32:
      __ RecordComment(
          "[  Mod32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Mod32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvModU32:
      __ RecordComment(
          "[  Modu32(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Modu32(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvMul64:
      __ RecordComment(
          "[  Mul64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Mul64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvDiv64: {
      __ RecordComment(
          "[  Div64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Div64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      // Set ouput to zero if divisor == 0
      __ RecordComment(
          "[  LoadZeroIfConditionZero(i.OutputRegister(), "
          "i.InputRegister(1));");
      __ LoadZeroIfConditionZero(i.OutputRegister(), i.InputRegister(1));
      __ RecordComment("]");
      break;
    }
    case kRiscvDivU64: {
      __ RecordComment(
          "[  Divu64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Divu64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      // Set ouput to zero if divisor == 0
      __ RecordComment(
          "[  LoadZeroIfConditionZero(i.OutputRegister(), "
          "i.InputRegister(1));");
      __ LoadZeroIfConditionZero(i.OutputRegister(), i.InputRegister(1));
      __ RecordComment("]");
      break;
    }
    case kRiscvMod64:
      __ RecordComment(
          "[  Mod64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Mod64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvModU64:
      __ RecordComment(
          "[  Modu64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Modu64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvAnd:
      __ RecordComment(
          "[  And(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));");
      __ And(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvAnd32:
      __ RecordComment(
          "[  And(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));");
      __ And(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      __ RecordComment(
          "[  Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);");
      __ Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);
      __ RecordComment("]");
      break;
    case kRiscvOr:
      __ RecordComment(
          "[  Or(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));");
      __ Or(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvOr32:
      __ RecordComment(
          "[  Or(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));");
      __ Or(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      __ RecordComment(
          "[  Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);");
      __ Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);
      __ RecordComment("]");
      break;
    case kRiscvNor:
      if (instr->InputAt(1)->IsRegister()) {
        __ RecordComment(
            "[  Nor(i.OutputRegister(), i.InputRegister(0), "
            "i.InputOperand(1));");
        __ Nor(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
        __ RecordComment("]");
      } else {
        DCHECK_EQ(0, i.InputOperand(1).immediate());
        __ RecordComment(
            "[  Nor(i.OutputRegister(), i.InputRegister(0), zero_reg);");
        __ Nor(i.OutputRegister(), i.InputRegister(0), zero_reg);
        __ RecordComment("]");
      }
      break;
    case kRiscvNor32:
      if (instr->InputAt(1)->IsRegister()) {
        __ RecordComment(
            "[  Nor(i.OutputRegister(), i.InputRegister(0), "
            "i.InputOperand(1));");
        __ Nor(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
        __ RecordComment("]");
        __ RecordComment(
            "[  Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);");
        __ Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);
        __ RecordComment("]");
      } else {
        DCHECK_EQ(0, i.InputOperand(1).immediate());
        __ RecordComment(
            "[  Nor(i.OutputRegister(), i.InputRegister(0), zero_reg);");
        __ Nor(i.OutputRegister(), i.InputRegister(0), zero_reg);
        __ RecordComment("]");
        __ RecordComment(
            "[  Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);");
        __ Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);
        __ RecordComment("]");
      }
      break;
    case kRiscvXor:
      __ RecordComment(
          "[  Xor(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));");
      __ Xor(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvXor32:
      __ RecordComment(
          "[  Xor(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));");
      __ Xor(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      __ RecordComment(
          "[  Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);");
      __ Sll32(i.OutputRegister(), i.OutputRegister(), 0x0);
      __ RecordComment("]");
      break;
    case kRiscvClz32:
      __ RecordComment("[  Clz32(i.OutputRegister(), i.InputRegister(0));");
      __ Clz32(i.OutputRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvClz64:
      __ RecordComment("[  Clz64(i.OutputRegister(), i.InputRegister(0));");
      __ Clz64(i.OutputRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvCtz32: {
      Register src = i.InputRegister(0);
      Register dst = i.OutputRegister();
      __ RecordComment("[  Ctz32(dst, src);");
      __ Ctz32(dst, src);
      __ RecordComment("]");
    } break;
    case kRiscvCtz64: {
      Register src = i.InputRegister(0);
      Register dst = i.OutputRegister();
      __ RecordComment("[  Ctz64(dst, src);");
      __ Ctz64(dst, src);
      __ RecordComment("]");
    } break;
    case kRiscvPopcnt32: {
      Register src = i.InputRegister(0);
      Register dst = i.OutputRegister();
      __ RecordComment("[  Popcnt32(dst, src);");
      __ Popcnt32(dst, src);
      __ RecordComment("]");
    } break;
    case kRiscvPopcnt64: {
      Register src = i.InputRegister(0);
      Register dst = i.OutputRegister();
      __ RecordComment("[  Popcnt64(dst, src);");
      __ Popcnt64(dst, src);
      __ RecordComment("]");
    } break;
    case kRiscvShl32:
      if (instr->InputAt(1)->IsRegister()) {
        __ RecordComment(
            "[  Sll32(i.OutputRegister(), i.InputRegister(0), "
            "i.InputRegister(1));");
        __ Sll32(i.OutputRegister(), i.InputRegister(0), i.InputRegister(1));
        __ RecordComment("]");
      } else {
        int64_t imm = i.InputOperand(1).immediate();
        __ Sll32(i.OutputRegister(), i.InputRegister(0),
                 static_cast<uint16_t>(imm));
      }
      break;
    case kRiscvShr32:
      if (instr->InputAt(1)->IsRegister()) {
        __ RecordComment(
            "[  Srl32(i.OutputRegister(), i.InputRegister(0), "
            "i.InputRegister(1));");
        __ Srl32(i.OutputRegister(), i.InputRegister(0), i.InputRegister(1));
        __ RecordComment("]");
      } else {
        int64_t imm = i.InputOperand(1).immediate();
        __ Srl32(i.OutputRegister(), i.InputRegister(0),
                 static_cast<uint16_t>(imm));
      }
      break;
    case kRiscvSar32:
      if (instr->InputAt(1)->IsRegister()) {
        __ RecordComment(
            "[  Sra32(i.OutputRegister(), i.InputRegister(0), "
            "i.InputRegister(1));");
        __ Sra32(i.OutputRegister(), i.InputRegister(0), i.InputRegister(1));
        __ RecordComment("]");
      } else {
        int64_t imm = i.InputOperand(1).immediate();
        __ Sra32(i.OutputRegister(), i.InputRegister(0),
                 static_cast<uint16_t>(imm));
      }
      break;
    case kRiscvZeroExtendWord: {
      __ RecordComment(
          "[  ZeroExtendWord(i.OutputRegister(), i.InputRegister(0));");
      __ ZeroExtendWord(i.OutputRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvSignExtendWord: {
      __ RecordComment(
          "[  SignExtendWord(i.OutputRegister(), i.InputRegister(0));");
      __ SignExtendWord(i.OutputRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvShl64:
      __ RecordComment(
          "[  Sll64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Sll64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvShr64:
      __ RecordComment(
          "[  Srl64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Srl64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvSar64:
      __ RecordComment(
          "[  Sra64(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Sra64(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvRor32:
      __ RecordComment(
          "[  Ror(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));");
      __ Ror(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvRor64:
      __ RecordComment(
          "[  Dror(i.OutputRegister(), i.InputRegister(0), "
          "i.InputOperand(1));");
      __ Dror(i.OutputRegister(), i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      break;
    case kRiscvTst:
      __ RecordComment(
          "[  And(kScratchReg, i.InputRegister(0), i.InputOperand(1));");
      __ And(kScratchReg, i.InputRegister(0), i.InputOperand(1));
      __ RecordComment("]");
      // Pseudo-instruction used for cmp/branch. No opcode emitted here.
      break;
    case kRiscvCmp:
      // Pseudo-instruction used for cmp/branch. No opcode emitted here.
      break;
    case kRiscvMov:
      // TODO(plind): Should we combine mov/li like this, or use separate instr?
      //    - Also see x64 ASSEMBLE_BINOP & RegisterOrOperandType
      if (HasRegisterInput(instr, 0)) {
        __ RecordComment("[  Move(i.OutputRegister(), i.InputRegister(0));");
        __ Move(i.OutputRegister(), i.InputRegister(0));
        __ RecordComment("]");
      } else {
        __ RecordComment("[  li(i.OutputRegister(), i.InputOperand(0));");
        __ li(i.OutputRegister(), i.InputOperand(0));
        __ RecordComment("]");
      }
      break;

    case kRiscvCmpS: {
      FPURegister left = i.InputOrZeroSingleRegister(0);
      FPURegister right = i.InputOrZeroSingleRegister(1);
      bool predicate;
      FPUCondition cc =
          FlagsConditionToConditionCmpFPU(&predicate, instr->flags_condition());

      if ((left == kDoubleRegZero || right == kDoubleRegZero) &&
          !__ IsSingleZeroRegSet()) {
        __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0f);");
        __ LoadFPRImmediate(kDoubleRegZero, 0.0f);
        __ RecordComment("]");
      }
      // compare result set to kScratchReg
      __ RecordComment("[  CompareF32(kScratchReg, cc, left, right);");
      __ CompareF32(kScratchReg, cc, left, right);
      __ RecordComment("]");
    } break;
    case kRiscvAddS:
      // TODO(plind): add special case: combine mult & add.
      __ fadd_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvSubS:
      __ fsub_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvMulS:
      // TODO(plind): add special case: right op is -1.0, see arm port.
      __ fmul_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvDivS:
      __ fdiv_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvModS: {
      // TODO(bmeurer): We should really get rid of this special instruction,
      // and generate a CallAddress instruction instead.
      FrameScope scope(tasm(), StackFrame::MANUAL);
      __ RecordComment("[  PrepareCallCFunction(0, 2, kScratchReg);");
      __ PrepareCallCFunction(0, 2, kScratchReg);
      __ RecordComment("]");
      __ MovToFloatParameters(i.InputDoubleRegister(0),
                              i.InputDoubleRegister(1));
      // TODO(balazs.kilvady): implement mod_two_floats_operation(isolate())
      __ RecordComment(
          "[  CallCFunction(ExternalReference::mod_two_doubles_operation(), 0, "
          "2);");
      __ CallCFunction(ExternalReference::mod_two_doubles_operation(), 0, 2);
      __ RecordComment("]");
      // Move the result in the double result register.
      __ RecordComment("[  MovFromFloatResult(i.OutputSingleRegister());");
      __ MovFromFloatResult(i.OutputSingleRegister());
      __ RecordComment("]");
      break;
    }
    case kRiscvAbsS:
      __ RecordComment(
          "[  fabs_s(i.OutputSingleRegister(), i.InputSingleRegister(0));");
      __ fabs_s(i.OutputSingleRegister(), i.InputSingleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvNegS:
      __ RecordComment(
          "[  Neg_s(i.OutputSingleRegister(), i.InputSingleRegister(0));");
      __ Neg_s(i.OutputSingleRegister(), i.InputSingleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvSqrtS: {
      __ RecordComment(
          "[  fsqrt_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0));");
      __ fsqrt_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvMaxS:
      __ fmax_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvMinS:
      __ fmin_s(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvCmpD: {
      FPURegister left = i.InputOrZeroDoubleRegister(0);
      FPURegister right = i.InputOrZeroDoubleRegister(1);
      bool predicate;
      FPUCondition cc =
          FlagsConditionToConditionCmpFPU(&predicate, instr->flags_condition());
      if ((left == kDoubleRegZero || right == kDoubleRegZero) &&
          !__ IsDoubleZeroRegSet()) {
        __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0);");
        __ LoadFPRImmediate(kDoubleRegZero, 0.0);
        __ RecordComment("]");
      }
      // compare result set to kScratchReg
      __ RecordComment("[  CompareF64(kScratchReg, cc, left, right);");
      __ CompareF64(kScratchReg, cc, left, right);
      __ RecordComment("]");
    } break;
    case kRiscvAddD:
      // TODO(plind): add special case: combine mult & add.
      __ fadd_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvSubD:
      __ fsub_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvMulD:
      // TODO(plind): add special case: right op is -1.0, see arm port.
      __ fmul_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvDivD:
      __ fdiv_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvModD: {
      // TODO(bmeurer): We should really get rid of this special instruction,
      // and generate a CallAddress instruction instead.
      FrameScope scope(tasm(), StackFrame::MANUAL);
      __ RecordComment("[  PrepareCallCFunction(0, 2, kScratchReg);");
      __ PrepareCallCFunction(0, 2, kScratchReg);
      __ RecordComment("]");
      __ MovToFloatParameters(i.InputDoubleRegister(0),
                              i.InputDoubleRegister(1));
      __ RecordComment(
          "[  CallCFunction(ExternalReference::mod_two_doubles_operation(), 0, "
          "2);");
      __ CallCFunction(ExternalReference::mod_two_doubles_operation(), 0, 2);
      __ RecordComment("]");
      // Move the result in the double result register.
      __ RecordComment("[  MovFromFloatResult(i.OutputDoubleRegister());");
      __ MovFromFloatResult(i.OutputDoubleRegister());
      __ RecordComment("]");
      break;
    }
    case kRiscvAbsD:
      __ RecordComment(
          "[  fabs_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0));");
      __ fabs_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvNegD:
      __ RecordComment(
          "[  Neg_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0));");
      __ Neg_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvSqrtD: {
      __ RecordComment(
          "[  fsqrt_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0));");
      __ fsqrt_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvMaxD:
      __ fmax_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvMinD:
      __ fmin_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                i.InputDoubleRegister(1));
      break;
    case kRiscvFloat64RoundDown: {
      __ Floor_d_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                   kScratchDoubleReg);
      break;
    }
    case kRiscvFloat32RoundDown: {
      __ Floor_s_s(i.OutputSingleRegister(), i.InputSingleRegister(0),
                   kScratchDoubleReg);
      break;
    }
    case kRiscvFloat64RoundTruncate: {
      __ Trunc_d_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                   kScratchDoubleReg);
      break;
    }
    case kRiscvFloat32RoundTruncate: {
      __ Trunc_s_s(i.OutputSingleRegister(), i.InputSingleRegister(0),
                   kScratchDoubleReg);
      break;
    }
    case kRiscvFloat64RoundUp: {
      __ Ceil_d_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                  kScratchDoubleReg);
      break;
    }
    case kRiscvFloat32RoundUp: {
      __ Ceil_s_s(i.OutputSingleRegister(), i.InputSingleRegister(0),
                  kScratchDoubleReg);
      break;
    }
    case kRiscvFloat64RoundTiesEven: {
      __ Round_d_d(i.OutputDoubleRegister(), i.InputDoubleRegister(0),
                   kScratchDoubleReg);
      break;
    }
    case kRiscvFloat32RoundTiesEven: {
      __ Round_s_s(i.OutputSingleRegister(), i.InputSingleRegister(0),
                   kScratchDoubleReg);
      break;
    }
    case kRiscvFloat32Max: {
      __ Float32Max(i.OutputSingleRegister(), i.InputSingleRegister(0),
                    i.InputSingleRegister(1));
      break;
    }
    case kRiscvFloat64Max: {
      __ Float64Max(i.OutputSingleRegister(), i.InputSingleRegister(0),
                    i.InputSingleRegister(1));
      break;
    }
    case kRiscvFloat32Min: {
      __ Float32Min(i.OutputSingleRegister(), i.InputSingleRegister(0),
                    i.InputSingleRegister(1));
      break;
    }
    case kRiscvFloat64Min: {
      __ Float64Min(i.OutputSingleRegister(), i.InputSingleRegister(0),
                    i.InputSingleRegister(1));
      break;
    }
    case kRiscvFloat64SilenceNaN:
      __ RecordComment(
          "[  FPUCanonicalizeNaN(i.OutputDoubleRegister(), "
          "i.InputDoubleRegister(0));");
      __ FPUCanonicalizeNaN(i.OutputDoubleRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvCvtSD:
      __ RecordComment(
          "[  fcvt_s_d(i.OutputSingleRegister(), i.InputDoubleRegister(0));");
      __ fcvt_s_d(i.OutputSingleRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvCvtDS:
      __ RecordComment(
          "[  fcvt_d_s(i.OutputDoubleRegister(), i.InputSingleRegister(0));");
      __ fcvt_d_s(i.OutputDoubleRegister(), i.InputSingleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvCvtDW: {
      __ RecordComment(
          "[  fcvt_d_w(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ fcvt_d_w(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvCvtSW: {
      __ RecordComment(
          "[  fcvt_s_w(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ fcvt_s_w(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvCvtSUw: {
      __ RecordComment(
          "[  Cvt_s_uw(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ Cvt_s_uw(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvCvtSL: {
      __ RecordComment(
          "[  fcvt_s_l(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ fcvt_s_l(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvCvtDL: {
      __ RecordComment(
          "[  fcvt_d_l(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ fcvt_d_l(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvCvtDUw: {
      __ RecordComment(
          "[  Cvt_d_uw(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ Cvt_d_uw(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvCvtDUl: {
      __ RecordComment(
          "[  Cvt_d_ul(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ Cvt_d_ul(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvCvtSUl: {
      __ RecordComment(
          "[  Cvt_s_ul(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ Cvt_s_ul(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    }
    case kRiscvFloorWD: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Floor_w_d(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Floor_w_d(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvCeilWD: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Ceil_w_d(i.OutputRegister(), i.InputDoubleRegister(0), result);");
      __ Ceil_w_d(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvRoundWD: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Round_w_d(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Round_w_d(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncWD: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_w_d(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_w_d(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvFloorWS: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Floor_w_s(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Floor_w_s(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvCeilWS: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Ceil_w_s(i.OutputRegister(), i.InputDoubleRegister(0), result);");
      __ Ceil_w_s(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvRoundWS: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Round_w_s(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Round_w_s(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncWS: {
      Label done;
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_w_s(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_w_s(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");

      // On RISCV, if the input value exceeds INT32_MAX, the result of fcvt
      // is INT32_MAX. Note that, since INT32_MAX means the lower 31-bits are
      // all 1s, INT32_MAX cannot be represented precisely as a float, so an
      // fcvt result of INT32_MAX always indicate overflow.
      //
      // In wasm_compiler, to detect overflow in converting a FP value, fval, to
      // integer, V8 checks whether I2F(F2I(fval)) equals fval. However, if fval
      // == INT32_MAX+1, the value of I2F(F2I(fval)) happens to be fval. So,
      // INT32_MAX is not a good value to indicate overflow. Instead, we will
      // use INT32_MIN as the converted result of an out-of-range FP value,
      // exploiting the fact that INT32_MAX+1 is INT32_MIN.
      //
      // If the result of conversion overflow, the result will be set to
      // INT32_MIN. Here we detect overflow by testing whether output + 1 <
      // output (i.e., kScratchReg  < output)
      __ RecordComment("[  Add32(kScratchReg, i.OutputRegister(), 1);");
      __ Add32(kScratchReg, i.OutputRegister(), 1);
      __ RecordComment("]");
      __ RecordComment(
          "[  Branch(&done, lt, i.OutputRegister(), Operand(kScratchReg));");
      __ Branch(&done, lt, i.OutputRegister(), Operand(kScratchReg));
      __ RecordComment("]");
      __ RecordComment("[  Move(i.OutputRegister(), kScratchReg);");
      __ Move(i.OutputRegister(), kScratchReg);
      __ RecordComment("]");
      __ RecordComment("[  bind(&done);");
      __ bind(&done);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncLS: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_l_s(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_l_s(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncLD: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_l_d(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_l_d(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncUwD: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_uw_d(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_uw_d(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncUwS: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_uw_s(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_uw_s(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");

      // On RISCV, if the input value exceeds UINT32_MAX, the result of fcvt
      // is UINT32_MAX. Note that, since UINT32_MAX means all 32-bits are 1s,
      // UINT32_MAX cannot be represented precisely as float, so an fcvt result
      // of UINT32_MAX always indicates overflow.
      //
      // In wasm_compiler.cc, to detect overflow in converting a FP value, fval,
      // to integer, V8 checks whether I2F(F2I(fval)) equals fval. However, if
      // fval == UINT32_MAX+1, the value of I2F(F2I(fval)) happens to be fval.
      // So, UINT32_MAX is not a good value to indicate overflow. Instead, we
      // will use 0 as the converted result of an out-of-range FP value,
      // exploiting the fact that UINT32_MAX+1 is 0.
      __ RecordComment("[  Add32(kScratchReg, i.OutputRegister(), 1);");
      __ Add32(kScratchReg, i.OutputRegister(), 1);
      __ RecordComment("]");
      // Set ouput to zero if result overflows (i.e., UINT32_MAX)
      __ RecordComment(
          "[  LoadZeroIfConditionZero(i.OutputRegister(), kScratchReg);");
      __ LoadZeroIfConditionZero(i.OutputRegister(), kScratchReg);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncUlS: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_ul_s(i.OutputRegister(), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_ul_s(i.OutputRegister(), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvTruncUlD: {
      Register result = instr->OutputCount() > 1 ? i.OutputRegister(1) : no_reg;
      __ RecordComment(
          "[  Trunc_ul_d(i.OutputRegister(0), i.InputDoubleRegister(0), "
          "result);");
      __ Trunc_ul_d(i.OutputRegister(0), i.InputDoubleRegister(0), result);
      __ RecordComment("]");
      break;
    }
    case kRiscvBitcastDL:
      __ RecordComment(
          "[  fmv_x_d(i.OutputRegister(), i.InputDoubleRegister(0));");
      __ fmv_x_d(i.OutputRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvBitcastLD:
      __ RecordComment(
          "[  fmv_d_x(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ fmv_d_x(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvBitcastInt32ToFloat32:
      __ RecordComment(
          "[  fmv_w_x(i.OutputDoubleRegister(), i.InputRegister(0));");
      __ fmv_w_x(i.OutputDoubleRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvBitcastFloat32ToInt32:
      __ RecordComment(
          "[  fmv_x_w(i.OutputRegister(), i.InputDoubleRegister(0));");
      __ fmv_x_w(i.OutputRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvFloat64ExtractLowWord32:
      __ RecordComment(
          "[  ExtractLowWordFromF64(i.OutputRegister(), "
          "i.InputDoubleRegister(0));");
      __ ExtractLowWordFromF64(i.OutputRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvFloat64ExtractHighWord32:
      __ RecordComment(
          "[  ExtractHighWordFromF64(i.OutputRegister(), "
          "i.InputDoubleRegister(0));");
      __ ExtractHighWordFromF64(i.OutputRegister(), i.InputDoubleRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvFloat64InsertLowWord32:
      __ RecordComment(
          "[  InsertLowWordF64(i.OutputDoubleRegister(), i.InputRegister(1));");
      __ InsertLowWordF64(i.OutputDoubleRegister(), i.InputRegister(1));
      __ RecordComment("]");
      break;
    case kRiscvFloat64InsertHighWord32:
      __ RecordComment(
          "[  InsertHighWordF64(i.OutputDoubleRegister(), "
          "i.InputRegister(1));");
      __ InsertHighWordF64(i.OutputDoubleRegister(), i.InputRegister(1));
      __ RecordComment("]");
      break;
      // ... more basic instructions ...

    case kRiscvSignExtendByte:
      __ RecordComment(
          "[  SignExtendByte(i.OutputRegister(), i.InputRegister(0));");
      __ SignExtendByte(i.OutputRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvSignExtendShort:
      __ RecordComment(
          "[  SignExtendShort(i.OutputRegister(), i.InputRegister(0));");
      __ SignExtendShort(i.OutputRegister(), i.InputRegister(0));
      __ RecordComment("]");
      break;
    case kRiscvLbu:
      __ RecordComment("[  Lbu(i.OutputRegister(), i.MemoryOperand());");
      __ Lbu(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvLb:
      __ RecordComment("[  Lb(i.OutputRegister(), i.MemoryOperand());");
      __ Lb(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvSb:
      __ RecordComment("[  Sb(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ Sb(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvLhu:
      __ RecordComment("[  Lhu(i.OutputRegister(), i.MemoryOperand());");
      __ Lhu(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvUlhu:
      __ RecordComment("[  Ulhu(i.OutputRegister(), i.MemoryOperand());");
      __ Ulhu(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvLh:
      __ RecordComment("[  Lh(i.OutputRegister(), i.MemoryOperand());");
      __ Lh(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvUlh:
      __ RecordComment("[  Ulh(i.OutputRegister(), i.MemoryOperand());");
      __ Ulh(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvSh:
      __ RecordComment("[  Sh(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ Sh(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvUsh:
      __ RecordComment("[  Ush(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ Ush(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvLw:
      __ RecordComment("[  Lw(i.OutputRegister(), i.MemoryOperand());");
      __ Lw(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvUlw:
      __ RecordComment("[  Ulw(i.OutputRegister(), i.MemoryOperand());");
      __ Ulw(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvLwu:
      __ RecordComment("[  Lwu(i.OutputRegister(), i.MemoryOperand());");
      __ Lwu(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvUlwu:
      __ RecordComment("[  Ulwu(i.OutputRegister(), i.MemoryOperand());");
      __ Ulwu(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvLd:
      __ RecordComment("[  Ld(i.OutputRegister(), i.MemoryOperand());");
      __ Ld(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvUld:
      __ RecordComment("[  Uld(i.OutputRegister(), i.MemoryOperand());");
      __ Uld(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kRiscvSw:
      __ RecordComment("[  Sw(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ Sw(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvUsw:
      __ RecordComment("[  Usw(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ Usw(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvSd:
      __ RecordComment("[  Sd(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ Sd(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvUsd:
      __ RecordComment("[  Usd(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ Usd(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvLoadFloat: {
      __ RecordComment(
          "[  LoadFloat(i.OutputSingleRegister(), i.MemoryOperand());");
      __ LoadFloat(i.OutputSingleRegister(), i.MemoryOperand());
      __ RecordComment("]");
      break;
    }
    case kRiscvULoadFloat: {
      __ RecordComment(
          "[  ULoadFloat(i.OutputSingleRegister(), i.MemoryOperand());");
      __ ULoadFloat(i.OutputSingleRegister(), i.MemoryOperand());
      __ RecordComment("]");
      break;
    }
    case kRiscvStoreFloat: {
      size_t index = 0;
      MemOperand operand = i.MemoryOperand(&index);
      FPURegister ft = i.InputOrZeroSingleRegister(index);
      if (ft == kDoubleRegZero && !__ IsSingleZeroRegSet()) {
        __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0f);");
        __ LoadFPRImmediate(kDoubleRegZero, 0.0f);
        __ RecordComment("]");
      }
      __ RecordComment("[  StoreFloat(ft, operand);");
      __ StoreFloat(ft, operand);
      __ RecordComment("]");
      break;
    }
    case kRiscvUStoreFloat: {
      size_t index = 0;
      MemOperand operand = i.MemoryOperand(&index);
      FPURegister ft = i.InputOrZeroSingleRegister(index);
      if (ft == kDoubleRegZero && !__ IsSingleZeroRegSet()) {
        __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0f);");
        __ LoadFPRImmediate(kDoubleRegZero, 0.0f);
        __ RecordComment("]");
      }
      __ RecordComment("[  UStoreFloat(ft, operand);");
      __ UStoreFloat(ft, operand);
      __ RecordComment("]");
      break;
    }
    case kRiscvLoadDouble:
      __ RecordComment(
          "[  LoadDouble(i.OutputDoubleRegister(), i.MemoryOperand());");
      __ LoadDouble(i.OutputDoubleRegister(), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvULoadDouble:
      __ RecordComment(
          "[  ULoadDouble(i.OutputDoubleRegister(), i.MemoryOperand());");
      __ ULoadDouble(i.OutputDoubleRegister(), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvStoreDouble: {
      FPURegister ft = i.InputOrZeroDoubleRegister(2);
      if (ft == kDoubleRegZero && !__ IsDoubleZeroRegSet()) {
        __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0);");
        __ LoadFPRImmediate(kDoubleRegZero, 0.0);
        __ RecordComment("]");
      }
      __ RecordComment("[  StoreDouble(ft, i.MemoryOperand());");
      __ StoreDouble(ft, i.MemoryOperand());
      __ RecordComment("]");
      break;
    }
    case kRiscvUStoreDouble: {
      FPURegister ft = i.InputOrZeroDoubleRegister(2);
      if (ft == kDoubleRegZero && !__ IsDoubleZeroRegSet()) {
        __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0);");
        __ LoadFPRImmediate(kDoubleRegZero, 0.0);
        __ RecordComment("]");
      }
      __ RecordComment("[  UStoreDouble(ft, i.MemoryOperand());");
      __ UStoreDouble(ft, i.MemoryOperand());
      __ RecordComment("]");
      break;
    }
    case kRiscvSync: {
      __ RecordComment("[  sync();");
      __ sync();
      __ RecordComment("]");
      break;
    }
    case kRiscvPush:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ RecordComment(
            "[  StoreDouble(i.InputDoubleRegister(0), MemOperand(sp, "
            "-kDoubleSize));");
        __ StoreDouble(i.InputDoubleRegister(0), MemOperand(sp, -kDoubleSize));
        __ RecordComment("]");
        __ RecordComment("[  Sub32(sp, sp, Operand(kDoubleSize));");
        __ Sub32(sp, sp, Operand(kDoubleSize));
        __ RecordComment("]");
        frame_access_state()->IncreaseSPDelta(kDoubleSize / kSystemPointerSize);
      } else {
        __ RecordComment("[  Push(i.InputRegister(0));");
        __ Push(i.InputRegister(0));
        __ RecordComment("]");
        frame_access_state()->IncreaseSPDelta(1);
      }
      break;
    case kRiscvPeek: {
      int reverse_slot = i.InputInt32(0);
      int offset =
          FrameSlotToFPOffset(frame()->GetTotalFrameSlotCount() - reverse_slot);
      if (instr->OutputAt(0)->IsFPRegister()) {
        LocationOperand* op = LocationOperand::cast(instr->OutputAt(0));
        if (op->representation() == MachineRepresentation::kFloat64) {
          __ RecordComment(
              "[  LoadDouble(i.OutputDoubleRegister(), MemOperand(fp, "
              "offset));");
          __ LoadDouble(i.OutputDoubleRegister(), MemOperand(fp, offset));
          __ RecordComment("]");
        } else {
          DCHECK_EQ(op->representation(), MachineRepresentation::kFloat32);
          __ LoadFloat(
              i.OutputSingleRegister(0),
              MemOperand(fp, offset + kLessSignificantWordInDoublewordOffset));
        }
      } else {
        __ RecordComment("[  Ld(i.OutputRegister(0), MemOperand(fp, offset));");
        __ Ld(i.OutputRegister(0), MemOperand(fp, offset));
        __ RecordComment("]");
      }
      break;
    }
    case kRiscvStackClaim: {
      __ RecordComment("[  Sub64(sp, sp, Operand(i.InputInt32(0)));");
      __ Sub64(sp, sp, Operand(i.InputInt32(0)));
      __ RecordComment("]");
      frame_access_state()->IncreaseSPDelta(i.InputInt32(0) /
                                            kSystemPointerSize);
      break;
    }
    case kRiscvStoreToStackSlot: {
      if (instr->InputAt(0)->IsFPRegister()) {
        if (instr->InputAt(0)->IsSimd128Register()) {
          UNREACHABLE();
        } else {
          __ StoreDouble(i.InputDoubleRegister(0),
                         MemOperand(sp, i.InputInt32(1)));
        }
      } else {
        __ RecordComment(
            "[  Sd(i.InputRegister(0), MemOperand(sp, i.InputInt32(1)));");
        __ Sd(i.InputRegister(0), MemOperand(sp, i.InputInt32(1)));
        __ RecordComment("]");
      }
      break;
    }
    case kRiscvByteSwap64: {
      __ RecordComment(
          "[  ByteSwap(i.OutputRegister(0), i.InputRegister(0), 8);");
      __ ByteSwap(i.OutputRegister(0), i.InputRegister(0), 8);
      __ RecordComment("]");
      break;
    }
    case kRiscvByteSwap32: {
      __ RecordComment(
          "[  ByteSwap(i.OutputRegister(0), i.InputRegister(0), 4);");
      __ ByteSwap(i.OutputRegister(0), i.InputRegister(0), 4);
      __ RecordComment("]");
      break;
    }
    case kWord32AtomicLoadInt8:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lb);
      break;
    case kWord32AtomicLoadUint8:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lbu);
      break;
    case kWord32AtomicLoadInt16:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lh);
      break;
    case kWord32AtomicLoadUint16:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lhu);
      break;
    case kWord32AtomicLoadWord32:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lw);
      break;
    case kRiscvWord64AtomicLoadUint8:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lbu);
      break;
    case kRiscvWord64AtomicLoadUint16:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lhu);
      break;
    case kRiscvWord64AtomicLoadUint32:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Lwu);
      break;
    case kRiscvWord64AtomicLoadUint64:
      ASSEMBLE_ATOMIC_LOAD_INTEGER(Ld);
      break;
    case kWord32AtomicStoreWord8:
      ASSEMBLE_ATOMIC_STORE_INTEGER(Sb);
      break;
    case kWord32AtomicStoreWord16:
      ASSEMBLE_ATOMIC_STORE_INTEGER(Sh);
      break;
    case kWord32AtomicStoreWord32:
      ASSEMBLE_ATOMIC_STORE_INTEGER(Sw);
      break;
    case kRiscvWord64AtomicStoreWord8:
      ASSEMBLE_ATOMIC_STORE_INTEGER(Sb);
      break;
    case kRiscvWord64AtomicStoreWord16:
      ASSEMBLE_ATOMIC_STORE_INTEGER(Sh);
      break;
    case kRiscvWord64AtomicStoreWord32:
      ASSEMBLE_ATOMIC_STORE_INTEGER(Sw);
      break;
    case kRiscvWord64AtomicStoreWord64:
      ASSEMBLE_ATOMIC_STORE_INTEGER(Sd);
      break;
    case kWord32AtomicExchangeInt8:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(Ll, Sc, true, 8, 32);
      break;
    case kWord32AtomicExchangeUint8:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(Ll, Sc, false, 8, 32);
      break;
    case kWord32AtomicExchangeInt16:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(Ll, Sc, true, 16, 32);
      break;
    case kWord32AtomicExchangeUint16:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(Ll, Sc, false, 16, 32);
      break;
    case kWord32AtomicExchangeWord32:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER(Ll, Sc);
      break;
    case kRiscvWord64AtomicExchangeUint8:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(Lld, Scd, false, 8, 64);
      break;
    case kRiscvWord64AtomicExchangeUint16:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(Lld, Scd, false, 16, 64);
      break;
    case kRiscvWord64AtomicExchangeUint32:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT(Lld, Scd, false, 32, 64);
      break;
    case kRiscvWord64AtomicExchangeUint64:
      ASSEMBLE_ATOMIC_EXCHANGE_INTEGER(Lld, Scd);
      break;
    case kWord32AtomicCompareExchangeInt8:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(Ll, Sc, true, 8, 32);
      break;
    case kWord32AtomicCompareExchangeUint8:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(Ll, Sc, false, 8, 32);
      break;
    case kWord32AtomicCompareExchangeInt16:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(Ll, Sc, true, 16, 32);
      break;
    case kWord32AtomicCompareExchangeUint16:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(Ll, Sc, false, 16, 32);
      break;
    case kWord32AtomicCompareExchangeWord32:
      __ RecordComment("[  Sll32(i.InputRegister(2), i.InputRegister(2), 0);");
      __ Sll32(i.InputRegister(2), i.InputRegister(2), 0);
      __ RecordComment("]");
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER(Ll, Sc);
      break;
    case kRiscvWord64AtomicCompareExchangeUint8:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(Lld, Scd, false, 8, 64);
      break;
    case kRiscvWord64AtomicCompareExchangeUint16:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(Lld, Scd, false, 16, 64);
      break;
    case kRiscvWord64AtomicCompareExchangeUint32:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT(Lld, Scd, false, 32, 64);
      break;
    case kRiscvWord64AtomicCompareExchangeUint64:
      ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER(Lld, Scd);
      break;
    case kRiscvStoreCompressTagged:
      __ RecordComment(
          "[  StoreTaggedField(i.InputOrZeroRegister(2), i.MemoryOperand());");
      __ StoreTaggedField(i.InputOrZeroRegister(2), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvLoadDecompressTaggedSigned:
      __ RecordComment(
          "[  DecompressTaggedSigned(i.OutputRegister(), i.MemoryOperand());");
      __ DecompressTaggedSigned(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvLoadDecompressTaggedPointer:
      __ RecordComment(
          "[  DecompressTaggedPointer(i.OutputRegister(), i.MemoryOperand());");
      __ DecompressTaggedPointer(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      break;
    case kRiscvLoadDecompressAnyTagged:
      __ RecordComment(
          "[  DecompressAnyTagged(i.OutputRegister(), i.MemoryOperand());");
      __ DecompressAnyTagged(i.OutputRegister(), i.MemoryOperand());
      __ RecordComment("]");
      break;
#define ATOMIC_BINOP_CASE(op, inst)                         \
  case kWord32Atomic##op##Int8:                             \
    ASSEMBLE_ATOMIC_BINOP_EXT(Ll, Sc, true, 8, inst, 32);   \
    break;                                                  \
  case kWord32Atomic##op##Uint8:                            \
    ASSEMBLE_ATOMIC_BINOP_EXT(Ll, Sc, false, 8, inst, 32);  \
    break;                                                  \
  case kWord32Atomic##op##Int16:                            \
    ASSEMBLE_ATOMIC_BINOP_EXT(Ll, Sc, true, 16, inst, 32);  \
    break;                                                  \
  case kWord32Atomic##op##Uint16:                           \
    ASSEMBLE_ATOMIC_BINOP_EXT(Ll, Sc, false, 16, inst, 32); \
    break;                                                  \
  case kWord32Atomic##op##Word32:                           \
    ASSEMBLE_ATOMIC_BINOP(Ll, Sc, inst);                    \
    break;
      ATOMIC_BINOP_CASE(Add, Add32)
      ATOMIC_BINOP_CASE(Sub, Sub32)
      ATOMIC_BINOP_CASE(And, And)
      ATOMIC_BINOP_CASE(Or, Or)
      ATOMIC_BINOP_CASE(Xor, Xor)
#undef ATOMIC_BINOP_CASE
#define ATOMIC_BINOP_CASE(op, inst)                           \
  case kRiscvWord64Atomic##op##Uint8:                         \
    ASSEMBLE_ATOMIC_BINOP_EXT(Lld, Scd, false, 8, inst, 64);  \
    break;                                                    \
  case kRiscvWord64Atomic##op##Uint16:                        \
    ASSEMBLE_ATOMIC_BINOP_EXT(Lld, Scd, false, 16, inst, 64); \
    break;                                                    \
  case kRiscvWord64Atomic##op##Uint32:                        \
    ASSEMBLE_ATOMIC_BINOP_EXT(Lld, Scd, false, 32, inst, 64); \
    break;                                                    \
  case kRiscvWord64Atomic##op##Uint64:                        \
    ASSEMBLE_ATOMIC_BINOP(Lld, Scd, inst);                    \
    break;
      ATOMIC_BINOP_CASE(Add, Add64)
      ATOMIC_BINOP_CASE(Sub, Sub64)
      ATOMIC_BINOP_CASE(And, And)
      ATOMIC_BINOP_CASE(Or, Or)
      ATOMIC_BINOP_CASE(Xor, Xor)
#undef ATOMIC_BINOP_CASE
    case kRiscvAssertEqual:
      __ Assert(eq, static_cast<AbortReason>(i.InputOperand(2).immediate()),
                i.InputRegister(0), Operand(i.InputRegister(1)));
      break;

    default:
      UNIMPLEMENTED();
  }
  return kSuccess;
}  // NOLINT(readability/fn_size)

#define UNSUPPORTED_COND(opcode, condition)                                    \
  StdoutStream{} << "Unsupported " << #opcode << " condition: \"" << condition \
                 << "\"";                                                      \
  UNIMPLEMENTED();

void AssembleBranchToLabels(CodeGenerator* gen, TurboAssembler* tasm,
                            Instruction* instr, FlagsCondition condition,
                            Label* tlabel, Label* flabel, bool fallthru) {
#undef __
#define __ tasm->
  RiscvOperandConverter i(gen, instr);

  Condition cc = kNoCondition;
  // RISC-V does not have condition code flags, so compare and branch are
  // implemented differently than on the other arch's. The compare operations
  // emit riscv64 pseudo-instructions, which are handled here by branch
  // instructions that do the actual comparison. Essential that the input
  // registers to compare pseudo-op are not modified before this branch op, as
  // they are tested here.

  if (instr->arch_opcode() == kRiscvTst) {
    cc = FlagsConditionToConditionTst(condition);
    __ RecordComment("[  Branch(tlabel, cc, kScratchReg, Operand(zero_reg));");
    __ Branch(tlabel, cc, kScratchReg, Operand(zero_reg));
    __ RecordComment("]");
  } else if (instr->arch_opcode() == kRiscvAdd64 ||
             instr->arch_opcode() == kRiscvSub64) {
    cc = FlagsConditionToConditionOvf(condition);
    __ RecordComment("[  Sra64(kScratchReg, i.OutputRegister(), 32);");
    __ Sra64(kScratchReg, i.OutputRegister(), 32);
    __ RecordComment("]");
    __ RecordComment("[  Sra64(kScratchReg2, i.OutputRegister(), 31);");
    __ Sra64(kScratchReg2, i.OutputRegister(), 31);
    __ RecordComment("]");
    __ RecordComment(
        "[  Branch(tlabel, cc, kScratchReg2, Operand(kScratchReg));");
    __ Branch(tlabel, cc, kScratchReg2, Operand(kScratchReg));
    __ RecordComment("]");
  } else if (instr->arch_opcode() == kRiscvAddOvf64 ||
             instr->arch_opcode() == kRiscvSubOvf64) {
    switch (condition) {
      // Overflow occurs if overflow register is negative
      case kOverflow:
        __ RecordComment(
            "[  Branch(tlabel, lt, kScratchReg, Operand(zero_reg));");
        __ Branch(tlabel, lt, kScratchReg, Operand(zero_reg));
        __ RecordComment("]");
        break;
      case kNotOverflow:
        __ RecordComment(
            "[  Branch(tlabel, ge, kScratchReg, Operand(zero_reg));");
        __ Branch(tlabel, ge, kScratchReg, Operand(zero_reg));
        __ RecordComment("]");
        break;
      default:
        UNSUPPORTED_COND(instr->arch_opcode(), condition);
        break;
    }
  } else if (instr->arch_opcode() == kRiscvMulOvf32) {
    // Overflow occurs if overflow register is not zero
    switch (condition) {
      case kOverflow:
        __ RecordComment(
            "[  Branch(tlabel, ne, kScratchReg, Operand(zero_reg));");
        __ Branch(tlabel, ne, kScratchReg, Operand(zero_reg));
        __ RecordComment("]");
        break;
      case kNotOverflow:
        __ RecordComment(
            "[  Branch(tlabel, eq, kScratchReg, Operand(zero_reg));");
        __ Branch(tlabel, eq, kScratchReg, Operand(zero_reg));
        __ RecordComment("]");
        break;
      default:
        UNSUPPORTED_COND(kRiscvMulOvf32, condition);
        break;
    }
  } else if (instr->arch_opcode() == kRiscvCmp) {
    cc = FlagsConditionToConditionCmp(condition);
    __ RecordComment(
        "[  Branch(tlabel, cc, i.InputRegister(0), i.InputOperand(1));");
    __ Branch(tlabel, cc, i.InputRegister(0), i.InputOperand(1));
    __ RecordComment("]");
  } else if (instr->arch_opcode() == kArchStackPointerGreaterThan) {
    cc = FlagsConditionToConditionCmp(condition);
    Register lhs_register = sp;
    uint32_t offset;
    if (gen->ShouldApplyOffsetToStackCheck(instr, &offset)) {
      lhs_register = i.TempRegister(0);
      __ RecordComment("[  Sub64(lhs_register, sp, offset);");
      __ Sub64(lhs_register, sp, offset);
      __ RecordComment("]");
    }
    __ RecordComment(
        "[  Branch(tlabel, cc, lhs_register, Operand(i.InputRegister(0)));");
    __ Branch(tlabel, cc, lhs_register, Operand(i.InputRegister(0)));
    __ RecordComment("]");
  } else if (instr->arch_opcode() == kRiscvCmpS ||
             instr->arch_opcode() == kRiscvCmpD) {
    bool predicate;
    FlagsConditionToConditionCmpFPU(&predicate, condition);
    // floating-point compare result is set in kScratchReg
    if (predicate) {
      __ RecordComment("[  BranchTrueF(kScratchReg, tlabel);");
      __ BranchTrueF(kScratchReg, tlabel);
      __ RecordComment("]");
    } else {
      __ RecordComment("[  BranchFalseF(kScratchReg, tlabel);");
      __ BranchFalseF(kScratchReg, tlabel);
      __ RecordComment("]");
    }
  } else {
    PrintF("AssembleArchBranch Unimplemented arch_opcode: %d\n",
           instr->arch_opcode());
    UNIMPLEMENTED();
  }
  __ RecordComment("[  Branch(flabel);  // no fallthru to flabel.");
  if (!fallthru) __ Branch(flabel);  // no fallthru to flabel.
  __ RecordComment("]");
#undef __
#define __ tasm()->
}

// Assembles branches after an instruction.
void CodeGenerator::AssembleArchBranch(Instruction* instr, BranchInfo* branch) {
  Label* tlabel = branch->true_label;
  Label* flabel = branch->false_label;

  AssembleBranchToLabels(this, tasm(), instr, branch->condition, tlabel, flabel,
                         branch->fallthru);
}

void CodeGenerator::AssembleBranchPoisoning(FlagsCondition condition,
                                            Instruction* instr) {
  // TODO(jarin) Handle float comparisons (kUnordered[Not]Equal).
  if (condition == kUnorderedEqual || condition == kUnorderedNotEqual) {
    return;
  }

  RiscvOperandConverter i(this, instr);
  condition = NegateFlagsCondition(condition);

  switch (instr->arch_opcode()) {
    case kRiscvCmp: {
      __ CompareI(kScratchReg, i.InputRegister(0), i.InputOperand(1),
                  FlagsConditionToConditionCmp(condition));
      __ RecordComment(
          "[  LoadZeroIfConditionNotZero(kSpeculationPoisonRegister, "
          "kScratchReg);");
      __ LoadZeroIfConditionNotZero(kSpeculationPoisonRegister, kScratchReg);
      __ RecordComment("]");
    }
      return;
    case kRiscvTst: {
      switch (condition) {
        case kEqual:
          __ RecordComment(
              "[  LoadZeroIfConditionZero(kSpeculationPoisonRegister, "
              "kScratchReg);");
          __ LoadZeroIfConditionZero(kSpeculationPoisonRegister, kScratchReg);
          __ RecordComment("]");
          break;
        case kNotEqual:
          __ LoadZeroIfConditionNotZero(kSpeculationPoisonRegister,
                                        kScratchReg);
          break;
        default:
          UNREACHABLE();
      }
    }
      return;
    case kRiscvAdd64:
    case kRiscvSub64: {
      // Check for overflow creates 1 or 0 for result.
      __ RecordComment("[  Srl64(kScratchReg, i.OutputRegister(), 63);");
      __ Srl64(kScratchReg, i.OutputRegister(), 63);
      __ RecordComment("]");
      __ RecordComment("[  Srl32(kScratchReg2, i.OutputRegister(), 31);");
      __ Srl32(kScratchReg2, i.OutputRegister(), 31);
      __ RecordComment("]");
      __ RecordComment("[  Xor(kScratchReg2, kScratchReg, kScratchReg2);");
      __ Xor(kScratchReg2, kScratchReg, kScratchReg2);
      __ RecordComment("]");
      switch (condition) {
        case kOverflow:
          __ LoadZeroIfConditionNotZero(kSpeculationPoisonRegister,
                                        kScratchReg2);
          break;
        case kNotOverflow:
          __ RecordComment(
              "[  LoadZeroIfConditionZero(kSpeculationPoisonRegister, "
              "kScratchReg2);");
          __ LoadZeroIfConditionZero(kSpeculationPoisonRegister, kScratchReg2);
          __ RecordComment("]");
          break;
        default:
          UNSUPPORTED_COND(instr->arch_opcode(), condition);
      }
    }
      return;
    case kRiscvAddOvf64:
    case kRiscvSubOvf64: {
      // Overflow occurs if overflow register is negative
      __ RecordComment("[  Slt(kScratchReg2, kScratchReg, zero_reg);");
      __ Slt(kScratchReg2, kScratchReg, zero_reg);
      __ RecordComment("]");
      switch (condition) {
        case kOverflow:
          __ LoadZeroIfConditionNotZero(kSpeculationPoisonRegister,
                                        kScratchReg2);
          break;
        case kNotOverflow:
          __ RecordComment(
              "[  LoadZeroIfConditionZero(kSpeculationPoisonRegister, "
              "kScratchReg2);");
          __ LoadZeroIfConditionZero(kSpeculationPoisonRegister, kScratchReg2);
          __ RecordComment("]");
          break;
        default:
          UNSUPPORTED_COND(instr->arch_opcode(), condition);
      }
    }
      return;
    case kRiscvMulOvf32: {
      // Overflow occurs if overflow register is not zero
      switch (condition) {
        case kOverflow:
          __ LoadZeroIfConditionNotZero(kSpeculationPoisonRegister,
                                        kScratchReg);
          break;
        case kNotOverflow:
          __ RecordComment(
              "[  LoadZeroIfConditionZero(kSpeculationPoisonRegister, "
              "kScratchReg);");
          __ LoadZeroIfConditionZero(kSpeculationPoisonRegister, kScratchReg);
          __ RecordComment("]");
          break;
        default:
          UNSUPPORTED_COND(instr->arch_opcode(), condition);
      }
    }
      return;
    case kRiscvCmpS:
    case kRiscvCmpD: {
      bool predicate;
      FlagsConditionToConditionCmpFPU(&predicate, condition);
      if (predicate) {
        __ RecordComment(
            "[  LoadZeroIfConditionNotZero(kSpeculationPoisonRegister, "
            "kScratchReg);");
        __ LoadZeroIfConditionNotZero(kSpeculationPoisonRegister, kScratchReg);
        __ RecordComment("]");
      } else {
        __ RecordComment(
            "[  LoadZeroIfConditionZero(kSpeculationPoisonRegister, "
            "kScratchReg);");
        __ LoadZeroIfConditionZero(kSpeculationPoisonRegister, kScratchReg);
        __ RecordComment("]");
      }
    }
      return;
    default:
      UNREACHABLE();
  }
}

#undef UNSUPPORTED_COND

void CodeGenerator::AssembleArchDeoptBranch(Instruction* instr,
                                            BranchInfo* branch) {
  AssembleArchBranch(instr, branch);
}

void CodeGenerator::AssembleArchJump(RpoNumber target) {
  __ RecordComment("[  Branch(GetLabel(target));");
  if (!IsNextInAssemblyOrder(target)) __ Branch(GetLabel(target));
  __ RecordComment("]");
}

void CodeGenerator::AssembleArchTrap(Instruction* instr,
                                     FlagsCondition condition) {
  class OutOfLineTrap final : public OutOfLineCode {
   public:
    OutOfLineTrap(CodeGenerator* gen, Instruction* instr)
        : OutOfLineCode(gen), instr_(instr), gen_(gen) {}
    void Generate() final {
      RiscvOperandConverter i(gen_, instr_);
      TrapId trap_id =
          static_cast<TrapId>(i.InputInt32(instr_->InputCount() - 1));
      GenerateCallToTrap(trap_id);
    }

   private:
    void GenerateCallToTrap(TrapId trap_id) {
      if (trap_id == TrapId::kInvalid) {
        // We cannot test calls to the runtime in cctest/test-run-wasm.
        // Therefore we emit a call to C here instead of a call to the runtime.
        // We use the context register as the scratch register, because we do
        // not have a context here.
        __ RecordComment("[  PrepareCallCFunction(0, 0, cp);");
        __ PrepareCallCFunction(0, 0, cp);
        __ RecordComment("]");
        __ CallCFunction(
            ExternalReference::wasm_call_trap_callback_for_testing(), 0);
        __ RecordComment("[  LeaveFrame(StackFrame::WASM);");
        __ LeaveFrame(StackFrame::WASM);
        __ RecordComment("]");
        auto call_descriptor = gen_->linkage()->GetIncomingDescriptor();
        int pop_count =
            static_cast<int>(call_descriptor->StackParameterCount());
        pop_count += (pop_count & 1);  // align
        __ RecordComment("[  Drop(pop_count);");
        __ Drop(pop_count);
        __ RecordComment("]");
        __ RecordComment("[  Ret();");
        __ Ret();
        __ RecordComment("]");
      } else {
        gen_->AssembleSourcePosition(instr_);
        // A direct call to a wasm runtime stub defined in this module.
        // Just encode the stub index. This will be patched when the code
        // is added to the native module and copied into wasm code space.
        __ RecordComment(
            "[  Call(static_cast<Address>(trap_id), "
            "RelocInfo::WASM_STUB_CALL);");
        __ Call(static_cast<Address>(trap_id), RelocInfo::WASM_STUB_CALL);
        __ RecordComment("]");
        ReferenceMap* reference_map =
            gen_->zone()->New<ReferenceMap>(gen_->zone());
        gen_->RecordSafepoint(reference_map, Safepoint::kNoLazyDeopt);
        if (FLAG_debug_code) {
          __ RecordComment("[  stop();");
          __ stop();
          __ RecordComment("]");
        }
      }
    }
    Instruction* instr_;
    CodeGenerator* gen_;
  };
  auto ool = zone()->New<OutOfLineTrap>(this, instr);
  Label* tlabel = ool->entry();
  AssembleBranchToLabels(this, tasm(), instr, condition, tlabel, nullptr, true);
}

// Assembles boolean materializations after an instruction.
void CodeGenerator::AssembleArchBoolean(Instruction* instr,
                                        FlagsCondition condition) {
  RiscvOperandConverter i(this, instr);

  // Materialize a full 32-bit 1 or 0 value. The result register is always the
  // last output of the instruction.
  DCHECK_NE(0u, instr->OutputCount());
  Register result = i.OutputRegister(instr->OutputCount() - 1);
  Condition cc = kNoCondition;
  // RISC-V does not have condition code flags, so compare and branch are
  // implemented differently than on the other arch's. The compare operations
  // emit riscv64 pseudo-instructions, which are checked and handled here.

  if (instr->arch_opcode() == kRiscvTst) {
    cc = FlagsConditionToConditionTst(condition);
    if (cc == eq) {
      __ RecordComment("[  Sltu(result, kScratchReg, 1);");
      __ Sltu(result, kScratchReg, 1);
      __ RecordComment("]");
    } else {
      __ RecordComment("[  Sltu(result, zero_reg, kScratchReg);");
      __ Sltu(result, zero_reg, kScratchReg);
      __ RecordComment("]");
    }
    return;
  } else if (instr->arch_opcode() == kRiscvAdd64 ||
             instr->arch_opcode() == kRiscvSub64) {
    cc = FlagsConditionToConditionOvf(condition);
    // Check for overflow creates 1 or 0 for result.
    __ RecordComment("[  Srl64(kScratchReg, i.OutputRegister(), 63);");
    __ Srl64(kScratchReg, i.OutputRegister(), 63);
    __ RecordComment("]");
    __ RecordComment("[  Srl32(kScratchReg2, i.OutputRegister(), 31);");
    __ Srl32(kScratchReg2, i.OutputRegister(), 31);
    __ RecordComment("]");
    __ RecordComment("[  Xor(result, kScratchReg, kScratchReg2);");
    __ Xor(result, kScratchReg, kScratchReg2);
    __ RecordComment("]");
    if (cc == eq)  // Toggle result for not overflow.
      __ RecordComment("[  Xor(result, result, 1);");
    __ Xor(result, result, 1);
    __ RecordComment("]");
    return;
  } else if (instr->arch_opcode() == kRiscvAddOvf64 ||
             instr->arch_opcode() == kRiscvSubOvf64) {
    // Overflow occurs if overflow register is negative
    __ RecordComment("[  Slt(result, kScratchReg, zero_reg);");
    __ Slt(result, kScratchReg, zero_reg);
    __ RecordComment("]");
  } else if (instr->arch_opcode() == kRiscvMulOvf32) {
    // Overflow occurs if overflow register is not zero
    __ RecordComment("[  Sgtu(result, kScratchReg, zero_reg);");
    __ Sgtu(result, kScratchReg, zero_reg);
    __ RecordComment("]");
  } else if (instr->arch_opcode() == kRiscvCmp) {
    cc = FlagsConditionToConditionCmp(condition);
    switch (cc) {
      case eq:
      case ne: {
        Register left = i.InputRegister(0);
        Operand right = i.InputOperand(1);
        if (instr->InputAt(1)->IsImmediate()) {
          if (is_int12(-right.immediate())) {
            if (right.immediate() == 0) {
              if (cc == eq) {
                __ RecordComment("[  Sltu(result, left, 1);");
                __ Sltu(result, left, 1);
                __ RecordComment("]");
              } else {
                __ RecordComment("[  Sltu(result, zero_reg, left);");
                __ Sltu(result, zero_reg, left);
                __ RecordComment("]");
              }
            } else {
              __ RecordComment(
                  "[  Add64(result, left, Operand(-right.immediate()));");
              __ Add64(result, left, Operand(-right.immediate()));
              __ RecordComment("]");
              if (cc == eq) {
                __ RecordComment("[  Sltu(result, result, 1);");
                __ Sltu(result, result, 1);
                __ RecordComment("]");
              } else {
                __ RecordComment("[  Sltu(result, zero_reg, result);");
                __ Sltu(result, zero_reg, result);
                __ RecordComment("]");
              }
            }
          } else {
            if (is_uint12(right.immediate())) {
              __ RecordComment("[  Xor(result, left, right);");
              __ Xor(result, left, right);
              __ RecordComment("]");
            } else {
              __ RecordComment("[  li(kScratchReg, right);");
              __ li(kScratchReg, right);
              __ RecordComment("]");
              __ RecordComment("[  Xor(result, left, kScratchReg);");
              __ Xor(result, left, kScratchReg);
              __ RecordComment("]");
            }
            if (cc == eq) {
              __ RecordComment("[  Sltu(result, result, 1);");
              __ Sltu(result, result, 1);
              __ RecordComment("]");
            } else {
              __ RecordComment("[  Sltu(result, zero_reg, result);");
              __ Sltu(result, zero_reg, result);
              __ RecordComment("]");
            }
          }
        } else {
          __ RecordComment("[  Xor(result, left, right);");
          __ Xor(result, left, right);
          __ RecordComment("]");
          if (cc == eq) {
            __ RecordComment("[  Sltu(result, result, 1);");
            __ Sltu(result, result, 1);
            __ RecordComment("]");
          } else {
            __ RecordComment("[  Sltu(result, zero_reg, result);");
            __ Sltu(result, zero_reg, result);
            __ RecordComment("]");
          }
        }
      } break;
      case lt:
      case ge: {
        Register left = i.InputRegister(0);
        Operand right = i.InputOperand(1);
        __ RecordComment("[  Slt(result, left, right);");
        __ Slt(result, left, right);
        __ RecordComment("]");
        if (cc == ge) {
          __ RecordComment("[  Xor(result, result, 1);");
          __ Xor(result, result, 1);
          __ RecordComment("]");
        }
      } break;
      case gt:
      case le: {
        Register left = i.InputRegister(1);
        Operand right = i.InputOperand(0);
        __ RecordComment("[  Slt(result, left, right);");
        __ Slt(result, left, right);
        __ RecordComment("]");
        if (cc == le) {
          __ RecordComment("[  Xor(result, result, 1);");
          __ Xor(result, result, 1);
          __ RecordComment("]");
        }
      } break;
      case Uless:
      case Ugreater_equal: {
        Register left = i.InputRegister(0);
        Operand right = i.InputOperand(1);
        __ RecordComment("[  Sltu(result, left, right);");
        __ Sltu(result, left, right);
        __ RecordComment("]");
        if (cc == Ugreater_equal) {
          __ RecordComment("[  Xor(result, result, 1);");
          __ Xor(result, result, 1);
          __ RecordComment("]");
        }
      } break;
      case Ugreater:
      case Uless_equal: {
        Register left = i.InputRegister(1);
        Operand right = i.InputOperand(0);
        __ RecordComment("[  Sltu(result, left, right);");
        __ Sltu(result, left, right);
        __ RecordComment("]");
        if (cc == Uless_equal) {
          __ RecordComment("[  Xor(result, result, 1);");
          __ Xor(result, result, 1);
          __ RecordComment("]");
        }
      } break;
      default:
        UNREACHABLE();
    }
    return;
  } else if (instr->arch_opcode() == kRiscvCmpD ||
             instr->arch_opcode() == kRiscvCmpS) {
    FPURegister left = i.InputOrZeroDoubleRegister(0);
    FPURegister right = i.InputOrZeroDoubleRegister(1);
    if ((instr->arch_opcode() == kRiscvCmpD) &&
        (left == kDoubleRegZero || right == kDoubleRegZero) &&
        !__ IsDoubleZeroRegSet()) {
      __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0);");
      __ LoadFPRImmediate(kDoubleRegZero, 0.0);
      __ RecordComment("]");
    } else if ((instr->arch_opcode() == kRiscvCmpS) &&
               (left == kDoubleRegZero || right == kDoubleRegZero) &&
               !__ IsSingleZeroRegSet()) {
      __ RecordComment("[  LoadFPRImmediate(kDoubleRegZero, 0.0f);");
      __ LoadFPRImmediate(kDoubleRegZero, 0.0f);
      __ RecordComment("]");
    }
    bool predicate;
    FlagsConditionToConditionCmpFPU(&predicate, condition);
    // RISCV compare returns 0 or 1, do nothing when predicate; otherwise
    // toggle kScratchReg (i.e., 0 -> 1, 1 -> 0)
    if (predicate) {
      __ RecordComment("[  Move(result, kScratchReg);");
      __ Move(result, kScratchReg);
      __ RecordComment("]");
    } else {
      __ RecordComment("[  Xor(result, kScratchReg, 1);");
      __ Xor(result, kScratchReg, 1);
      __ RecordComment("]");
    }
    return;
  } else {
    PrintF("AssembleArchBranch Unimplemented arch_opcode is : %d\n",
           instr->arch_opcode());
    TRACE_UNIMPL();
    UNIMPLEMENTED();
  }
}

void CodeGenerator::AssembleArchBinarySearchSwitch(Instruction* instr) {
  RiscvOperandConverter i(this, instr);
  Register input = i.InputRegister(0);
  std::vector<std::pair<int32_t, Label*>> cases;
  for (size_t index = 2; index < instr->InputCount(); index += 2) {
    cases.push_back({i.InputInt32(index + 0), GetLabel(i.InputRpo(index + 1))});
  }
  AssembleArchBinarySearchSwitchRange(input, i.InputRpo(1), cases.data(),
                                      cases.data() + cases.size());
}

void CodeGenerator::AssembleArchTableSwitch(Instruction* instr) {
  RiscvOperandConverter i(this, instr);
  Register input = i.InputRegister(0);
  size_t const case_count = instr->InputCount() - 2;

  __ Branch(GetLabel(i.InputRpo(1)), Ugreater_equal, input,
            Operand(case_count));
  __ GenerateSwitchTable(input, case_count, [&i, this](size_t index) {
    return GetLabel(i.InputRpo(index + 2));
  });
}

void CodeGenerator::FinishFrame(Frame* frame) {
  auto call_descriptor = linkage()->GetIncomingDescriptor();

  const RegList saves_fpu = call_descriptor->CalleeSavedFPRegisters();
  if (saves_fpu != 0) {
    int count = base::bits::CountPopulation(saves_fpu);
    DCHECK_EQ(kNumCalleeSavedFPU, count);
    frame->AllocateSavedCalleeRegisterSlots(count *
                                            (kDoubleSize / kSystemPointerSize));
  }

  const RegList saves = call_descriptor->CalleeSavedRegisters();
  if (saves != 0) {
    int count = base::bits::CountPopulation(saves);
    DCHECK_EQ(kNumCalleeSaved, count + 1);
    frame->AllocateSavedCalleeRegisterSlots(count);
  }
}

void CodeGenerator::AssembleConstructFrame() {
  auto call_descriptor = linkage()->GetIncomingDescriptor();

  if (frame_access_state()->has_frame()) {
    if (call_descriptor->IsCFunctionCall()) {
      if (info()->GetOutputStackFrameType() == StackFrame::C_WASM_ENTRY) {
        __ RecordComment("[  StubPrologue(StackFrame::C_WASM_ENTRY);");
        __ StubPrologue(StackFrame::C_WASM_ENTRY);
        __ RecordComment("]");
        // Reserve stack space for saving the c_entry_fp later.
        __ RecordComment("[  Sub64(sp, sp, Operand(kSystemPointerSize));");
        __ Sub64(sp, sp, Operand(kSystemPointerSize));
        __ RecordComment("]");
      } else {
        __ RecordComment("[  Push(ra, fp);");
        __ Push(ra, fp);
        __ RecordComment("]");
        __ RecordComment("[  Move(fp, sp);");
        __ Move(fp, sp);
        __ RecordComment("]");
      }
    } else if (call_descriptor->IsJSFunctionCall()) {
      __ RecordComment("[  Prologue();");
      __ Prologue();
      __ RecordComment("]");
    } else {
      __ RecordComment("[  StubPrologue(info()->GetOutputStackFrameType());");
      __ StubPrologue(info()->GetOutputStackFrameType());
      __ RecordComment("]");
      if (call_descriptor->IsWasmFunctionCall()) {
        __ RecordComment("[  Push(kWasmInstanceRegister);");
        __ Push(kWasmInstanceRegister);
        __ RecordComment("]");
      } else if (call_descriptor->IsWasmImportWrapper() ||
                 call_descriptor->IsWasmCapiFunction()) {
        // Wasm import wrappers are passed a tuple in the place of the instance.
        // Unpack the tuple into the instance and the target callable.
        // This must be done here in the codegen because it cannot be expressed
        // properly in the graph.
        __ Ld(kJSFunctionRegister,
              FieldMemOperand(kWasmInstanceRegister, Tuple2::kValue2Offset));
        __ Ld(kWasmInstanceRegister,
              FieldMemOperand(kWasmInstanceRegister, Tuple2::kValue1Offset));
        __ RecordComment("[  Push(kWasmInstanceRegister);");
        __ Push(kWasmInstanceRegister);
        __ RecordComment("]");
        if (call_descriptor->IsWasmCapiFunction()) {
          // Reserve space for saving the PC later.
          __ RecordComment("[  Sub64(sp, sp, Operand(kSystemPointerSize));");
          __ Sub64(sp, sp, Operand(kSystemPointerSize));
          __ RecordComment("]");
        }
      }
    }
  }

  int required_slots =
      frame()->GetTotalFrameSlotCount() - frame()->GetFixedSlotCount();

  if (info()->is_osr()) {
    // TurboFan OSR-compiled functions cannot be entered directly.
    __ RecordComment(
        "[  Abort(AbortReason::kShouldNotDirectlyEnterOsrFunction);");
    __ Abort(AbortReason::kShouldNotDirectlyEnterOsrFunction);
    __ RecordComment("]");

    // Unoptimized code jumps directly to this entrypoint while the unoptimized
    // frame is still on the stack. Optimized code uses OSR values directly from
    // the unoptimized frame. Thus, all that needs to be done is to allocate the
    // remaining stack slots.
    if (FLAG_code_comments) __ RecordComment("-- OSR entrypoint --");
    __ RecordComment("[  pc_offset();");
    osr_pc_offset_ = __ pc_offset();
    __ RecordComment("]");
    required_slots -= osr_helper()->UnoptimizedFrameSlots();
    ResetSpeculationPoison();
  }

  const RegList saves = call_descriptor->CalleeSavedRegisters();
  const RegList saves_fpu = call_descriptor->CalleeSavedFPRegisters();

  if (required_slots > 0) {
    DCHECK(frame_access_state()->has_frame());
    if (info()->IsWasm() && required_slots > 128) {
      // For WebAssembly functions with big frames we have to do the stack
      // overflow check before we construct the frame. Otherwise we may not
      // have enough space on the stack to call the runtime for the stack
      // overflow.
      Label done;

      // If the frame is bigger than the stack, we throw the stack overflow
      // exception unconditionally. Thereby we can avoid the integer overflow
      // check in the condition code.
      if ((required_slots * kSystemPointerSize) < (FLAG_stack_size * 1024)) {
        __ Ld(
            kScratchReg,
            FieldMemOperand(kWasmInstanceRegister,
                            WasmInstanceObject::kRealStackLimitAddressOffset));
        __ RecordComment("[  Ld(kScratchReg, MemOperand(kScratchReg));");
        __ Ld(kScratchReg, MemOperand(kScratchReg));
        __ RecordComment("]");
        __ Add64(kScratchReg, kScratchReg,
                 Operand(required_slots * kSystemPointerSize));
        __ RecordComment("[  Branch(&done, uge, sp, Operand(kScratchReg));");
        __ Branch(&done, uge, sp, Operand(kScratchReg));
        __ RecordComment("]");
      }

      __ RecordComment(
          "[  Call(wasm::WasmCode::kWasmStackOverflow, "
          "RelocInfo::WASM_STUB_CALL);");
      __ Call(wasm::WasmCode::kWasmStackOverflow, RelocInfo::WASM_STUB_CALL);
      __ RecordComment("]");
      // We come from WebAssembly, there are no references for the GC.
      ReferenceMap* reference_map = zone()->New<ReferenceMap>(zone());
      RecordSafepoint(reference_map, Safepoint::kNoLazyDeopt);
      if (FLAG_debug_code) {
        __ RecordComment("[  stop();");
        __ stop();
        __ RecordComment("]");
      }

      __ RecordComment("[  bind(&done);");
      __ bind(&done);
      __ RecordComment("]");
    }
  }

  const int returns = frame()->GetReturnSlotCount();

  // Skip callee-saved and return slots, which are pushed below.
  required_slots -= base::bits::CountPopulation(saves);
  required_slots -= base::bits::CountPopulation(saves_fpu);
  required_slots -= returns;
  if (required_slots > 0) {
    __ RecordComment(
        "[  Sub64(sp, sp, Operand(required_slots * kSystemPointerSize));");
    __ Sub64(sp, sp, Operand(required_slots * kSystemPointerSize));
    __ RecordComment("]");
  }

  if (saves_fpu != 0) {
    // Save callee-saved FPU registers.
    __ RecordComment("[  MultiPushFPU(saves_fpu);");
    __ MultiPushFPU(saves_fpu);
    __ RecordComment("]");
    DCHECK_EQ(kNumCalleeSavedFPU, base::bits::CountPopulation(saves_fpu));
  }

  if (saves != 0) {
    // Save callee-saved registers.
    __ RecordComment("[  MultiPush(saves);");
    __ MultiPush(saves);
    __ RecordComment("]");
    DCHECK_EQ(kNumCalleeSaved, base::bits::CountPopulation(saves) + 1);
  }

  if (returns != 0) {
    // Create space for returns.
    __ RecordComment(
        "[  Sub64(sp, sp, Operand(returns * kSystemPointerSize));");
    __ Sub64(sp, sp, Operand(returns * kSystemPointerSize));
    __ RecordComment("]");
  }
}

void CodeGenerator::AssembleReturn(InstructionOperand* pop) {
  auto call_descriptor = linkage()->GetIncomingDescriptor();

  const int returns = frame()->GetReturnSlotCount();
  if (returns != 0) {
    __ RecordComment(
        "[  Add64(sp, sp, Operand(returns * kSystemPointerSize));");
    __ Add64(sp, sp, Operand(returns * kSystemPointerSize));
    __ RecordComment("]");
  }

  // Restore GP registers.
  const RegList saves = call_descriptor->CalleeSavedRegisters();
  if (saves != 0) {
    __ RecordComment("[  MultiPop(saves);");
    __ MultiPop(saves);
    __ RecordComment("]");
  }

  // Restore FPU registers.
  const RegList saves_fpu = call_descriptor->CalleeSavedFPRegisters();
  if (saves_fpu != 0) {
    __ RecordComment("[  MultiPopFPU(saves_fpu);");
    __ MultiPopFPU(saves_fpu);
    __ RecordComment("]");
  }

  RiscvOperandConverter g(this, nullptr);
  if (call_descriptor->IsCFunctionCall()) {
    AssembleDeconstructFrame();
  } else if (frame_access_state()->has_frame()) {
    // Canonicalize JSFunction return sites for now unless they have an variable
    // number of stack slot pops.
    if (pop->IsImmediate() && g.ToConstant(pop).ToInt32() == 0) {
      if (return_label_.is_bound()) {
        __ RecordComment("[  Branch(&return_label_);");
        __ Branch(&return_label_);
        __ RecordComment("]");
        return;
      } else {
        __ RecordComment("[  bind(&return_label_);");
        __ bind(&return_label_);
        __ RecordComment("]");
        AssembleDeconstructFrame();
      }
    } else {
      AssembleDeconstructFrame();
    }
  }
  int pop_count = static_cast<int>(call_descriptor->StackParameterCount());
  if (pop->IsImmediate()) {
    pop_count += g.ToConstant(pop).ToInt32();
  } else {
    Register pop_reg = g.ToRegister(pop);
    __ RecordComment("[  Sll64(pop_reg, pop_reg, kSystemPointerSizeLog2);");
    __ Sll64(pop_reg, pop_reg, kSystemPointerSizeLog2);
    __ RecordComment("]");
    __ RecordComment("[  Add64(sp, sp, pop_reg);");
    __ Add64(sp, sp, pop_reg);
    __ RecordComment("]");
  }
  if (pop_count != 0) {
    __ RecordComment("[  DropAndRet(pop_count);");
    __ DropAndRet(pop_count);
    __ RecordComment("]");
  } else {
    __ RecordComment("[  Ret();");
    __ Ret();
    __ RecordComment("]");
  }
}

void CodeGenerator::FinishCode() {}

void CodeGenerator::PrepareForDeoptimizationExits(int deopt_count) {}

void CodeGenerator::AssembleMove(InstructionOperand* source,
                                 InstructionOperand* destination) {
  RiscvOperandConverter g(this, nullptr);
  // Dispatch on the source and destination operand kinds.  Not all
  // combinations are possible.
  if (source->IsRegister()) {
    DCHECK(destination->IsRegister() || destination->IsStackSlot());
    Register src = g.ToRegister(source);
    if (destination->IsRegister()) {
      __ RecordComment("[  Move(g.ToRegister(destination), src);");
      __ Move(g.ToRegister(destination), src);
      __ RecordComment("]");
    } else {
      __ RecordComment("[  Sd(src, g.ToMemOperand(destination));");
      __ Sd(src, g.ToMemOperand(destination));
      __ RecordComment("]");
    }
  } else if (source->IsStackSlot()) {
    DCHECK(destination->IsRegister() || destination->IsStackSlot());
    MemOperand src = g.ToMemOperand(source);
    if (destination->IsRegister()) {
      __ RecordComment("[  Ld(g.ToRegister(destination), src);");
      __ Ld(g.ToRegister(destination), src);
      __ RecordComment("]");
    } else {
      Register temp = kScratchReg;
      __ RecordComment("[  Ld(temp, src);");
      __ Ld(temp, src);
      __ RecordComment("]");
      __ RecordComment("[  Sd(temp, g.ToMemOperand(destination));");
      __ Sd(temp, g.ToMemOperand(destination));
      __ RecordComment("]");
    }
  } else if (source->IsConstant()) {
    Constant src = g.ToConstant(source);
    if (destination->IsRegister() || destination->IsStackSlot()) {
      Register dst =
          destination->IsRegister() ? g.ToRegister(destination) : kScratchReg;
      switch (src.type()) {
        case Constant::kInt32:
          __ RecordComment("[  li(dst, Operand(src.ToInt32()));");
          __ li(dst, Operand(src.ToInt32()));
          __ RecordComment("]");
          break;
        case Constant::kFloat32:
          __ RecordComment(
              "[  li(dst, Operand::EmbeddedNumber(src.ToFloat32()));");
          __ li(dst, Operand::EmbeddedNumber(src.ToFloat32()));
          __ RecordComment("]");
          break;
        case Constant::kInt64:
          if (RelocInfo::IsWasmReference(src.rmode())) {
            __ RecordComment(
                "[  li(dst, Operand(src.ToInt64(), src.rmode()));");
            __ li(dst, Operand(src.ToInt64(), src.rmode()));
            __ RecordComment("]");
          } else {
            __ RecordComment("[  li(dst, Operand(src.ToInt64()));");
            __ li(dst, Operand(src.ToInt64()));
            __ RecordComment("]");
          }
          break;
        case Constant::kFloat64:
          __ RecordComment(
              "[  li(dst, Operand::EmbeddedNumber(src.ToFloat64().value()));");
          __ li(dst, Operand::EmbeddedNumber(src.ToFloat64().value()));
          __ RecordComment("]");
          break;
        case Constant::kExternalReference:
          __ RecordComment("[  li(dst, src.ToExternalReference());");
          __ li(dst, src.ToExternalReference());
          __ RecordComment("]");
          break;
        case Constant::kDelayedStringConstant:
          __ RecordComment("[  li(dst, src.ToDelayedStringConstant());");
          __ li(dst, src.ToDelayedStringConstant());
          __ RecordComment("]");
          break;
        case Constant::kHeapObject: {
          Handle<HeapObject> src_object = src.ToHeapObject();
          RootIndex index;
          if (IsMaterializableFromRoot(src_object, &index)) {
            __ RecordComment("[  LoadRoot(dst, index);");
            __ LoadRoot(dst, index);
            __ RecordComment("]");
          } else {
            __ RecordComment("[  li(dst, src_object);");
            __ li(dst, src_object);
            __ RecordComment("]");
          }
          break;
        }
        case Constant::kCompressedHeapObject: {
          Handle<HeapObject> src_object = src.ToHeapObject();
          RootIndex index;
          if (IsMaterializableFromRoot(src_object, &index)) {
            __ RecordComment("[  LoadRoot(dst, index);");
            __ LoadRoot(dst, index);
            __ RecordComment("]");
          } else {
            // TODO(v8:7703, jyan@ca.ibm.com): Turn into a
            // COMPRESSED_EMBEDDED_OBJECT when the constant pool entry size is
            // tagged size.
            __ RecordComment(
                "[  li(dst, src_object, RelocInfo::COMPRESSED_EMBEDDED_OBJECT);");
            __ li(dst, src_object, RelocInfo::COMPRESSED_EMBEDDED_OBJECT);
            __ RecordComment("]");
          }
          break;
        }
        case Constant::kRpoNumber:
          UNREACHABLE();  // TODO(titzer): loading RPO numbers
          break;
      }
      __ RecordComment("[  Sd(dst, g.ToMemOperand(destination));");
      if (destination->IsStackSlot()) __ Sd(dst, g.ToMemOperand(destination));
      __ RecordComment("]");
    } else if (src.type() == Constant::kFloat32) {
      if (destination->IsFPStackSlot()) {
        MemOperand dst = g.ToMemOperand(destination);
        if (bit_cast<int32_t>(src.ToFloat32()) == 0) {
          __ RecordComment("[  Sw(zero_reg, dst);");
          __ Sw(zero_reg, dst);
          __ RecordComment("]");
        } else {
          __ RecordComment(
              "[  li(kScratchReg, "
              "Operand(bit_cast<int32_t>(src.ToFloat32())));");
          __ li(kScratchReg, Operand(bit_cast<int32_t>(src.ToFloat32())));
          __ RecordComment("]");
          __ RecordComment("[  Sw(kScratchReg, dst);");
          __ Sw(kScratchReg, dst);
          __ RecordComment("]");
        }
      } else {
        DCHECK(destination->IsFPRegister());
        FloatRegister dst = g.ToSingleRegister(destination);
        __ RecordComment("[  LoadFPRImmediate(dst, src.ToFloat32());");
        __ LoadFPRImmediate(dst, src.ToFloat32());
        __ RecordComment("]");
      }
    } else {
      DCHECK_EQ(Constant::kFloat64, src.type());
      DoubleRegister dst = destination->IsFPRegister()
                               ? g.ToDoubleRegister(destination)
                               : kScratchDoubleReg;
      __ RecordComment("[  LoadFPRImmediate(dst, src.ToFloat64().value());");
      __ LoadFPRImmediate(dst, src.ToFloat64().value());
      __ RecordComment("]");
      if (destination->IsFPStackSlot()) {
        __ RecordComment("[  StoreDouble(dst, g.ToMemOperand(destination));");
        __ StoreDouble(dst, g.ToMemOperand(destination));
        __ RecordComment("]");
      }
    }
  } else if (source->IsFPRegister()) {
    MachineRepresentation rep = LocationOperand::cast(source)->representation();
    if (rep == MachineRepresentation::kSimd128) {
      UNIMPLEMENTED();
    } else {
      FPURegister src = g.ToDoubleRegister(source);
      if (destination->IsFPRegister()) {
        FPURegister dst = g.ToDoubleRegister(destination);
        __ RecordComment("[  Move(dst, src);");
        __ Move(dst, src);
        __ RecordComment("]");
      } else {
        DCHECK(destination->IsFPStackSlot());
        if (rep == MachineRepresentation::kFloat32) {
          __ RecordComment("[  StoreFloat(src, g.ToMemOperand(destination));");
          __ StoreFloat(src, g.ToMemOperand(destination));
          __ RecordComment("]");
        } else {
          DCHECK_EQ(rep, MachineRepresentation::kFloat64);
          __ RecordComment("[  StoreDouble(src, g.ToMemOperand(destination));");
          __ StoreDouble(src, g.ToMemOperand(destination));
          __ RecordComment("]");
        }
      }
    }
  } else if (source->IsFPStackSlot()) {
    DCHECK(destination->IsFPRegister() || destination->IsFPStackSlot());
    MemOperand src = g.ToMemOperand(source);
    MachineRepresentation rep = LocationOperand::cast(source)->representation();
    if (rep == MachineRepresentation::kSimd128) {
      UNIMPLEMENTED();
    } else {
      if (destination->IsFPRegister()) {
        if (rep == MachineRepresentation::kFloat32) {
          __ RecordComment(
              "[  LoadFloat(g.ToDoubleRegister(destination), src);");
          __ LoadFloat(g.ToDoubleRegister(destination), src);
          __ RecordComment("]");
        } else {
          DCHECK_EQ(rep, MachineRepresentation::kFloat64);
          __ RecordComment(
              "[  LoadDouble(g.ToDoubleRegister(destination), src);");
          __ LoadDouble(g.ToDoubleRegister(destination), src);
          __ RecordComment("]");
        }
      } else {
        DCHECK(destination->IsFPStackSlot());
        FPURegister temp = kScratchDoubleReg;
        if (rep == MachineRepresentation::kFloat32) {
          __ RecordComment("[  LoadFloat(temp, src);");
          __ LoadFloat(temp, src);
          __ RecordComment("]");
          __ RecordComment("[  StoreFloat(temp, g.ToMemOperand(destination));");
          __ StoreFloat(temp, g.ToMemOperand(destination));
          __ RecordComment("]");
        } else {
          DCHECK_EQ(rep, MachineRepresentation::kFloat64);
          __ RecordComment("[  LoadDouble(temp, src);");
          __ LoadDouble(temp, src);
          __ RecordComment("]");
          __ RecordComment(
              "[  StoreDouble(temp, g.ToMemOperand(destination));");
          __ StoreDouble(temp, g.ToMemOperand(destination));
          __ RecordComment("]");
        }
      }
    }
  } else {
    UNREACHABLE();
  }
}

void CodeGenerator::AssembleSwap(InstructionOperand* source,
                                 InstructionOperand* destination) {
  RiscvOperandConverter g(this, nullptr);
  // Dispatch on the source and destination operand kinds.  Not all
  // combinations are possible.
  if (source->IsRegister()) {
    // Register-register.
    Register temp = kScratchReg;
    Register src = g.ToRegister(source);
    if (destination->IsRegister()) {
      Register dst = g.ToRegister(destination);
      __ RecordComment("[  Move(temp, src);");
      __ Move(temp, src);
      __ RecordComment("]");
      __ RecordComment("[  Move(src, dst);");
      __ Move(src, dst);
      __ RecordComment("]");
      __ RecordComment("[  Move(dst, temp);");
      __ Move(dst, temp);
      __ RecordComment("]");
    } else {
      DCHECK(destination->IsStackSlot());
      MemOperand dst = g.ToMemOperand(destination);
      __ RecordComment("[  Move(temp, src);");
      __ Move(temp, src);
      __ RecordComment("]");
      __ RecordComment("[  Ld(src, dst);");
      __ Ld(src, dst);
      __ RecordComment("]");
      __ RecordComment("[  Sd(temp, dst);");
      __ Sd(temp, dst);
      __ RecordComment("]");
    }
  } else if (source->IsStackSlot()) {
    DCHECK(destination->IsStackSlot());
    Register temp_0 = kScratchReg;
    Register temp_1 = kScratchReg2;
    MemOperand src = g.ToMemOperand(source);
    MemOperand dst = g.ToMemOperand(destination);
    __ RecordComment("[  Ld(temp_0, src);");
    __ Ld(temp_0, src);
    __ RecordComment("]");
    __ RecordComment("[  Ld(temp_1, dst);");
    __ Ld(temp_1, dst);
    __ RecordComment("]");
    __ RecordComment("[  Sd(temp_0, dst);");
    __ Sd(temp_0, dst);
    __ RecordComment("]");
    __ RecordComment("[  Sd(temp_1, src);");
    __ Sd(temp_1, src);
    __ RecordComment("]");
  } else if (source->IsFPRegister()) {
    MachineRepresentation rep = LocationOperand::cast(source)->representation();
    if (rep == MachineRepresentation::kSimd128) {
      UNIMPLEMENTED();
    } else {
      FPURegister temp = kScratchDoubleReg;
      FPURegister src = g.ToDoubleRegister(source);
      if (destination->IsFPRegister()) {
        FPURegister dst = g.ToDoubleRegister(destination);
        __ RecordComment("[  Move(temp, src);");
        __ Move(temp, src);
        __ RecordComment("]");
        __ RecordComment("[  Move(src, dst);");
        __ Move(src, dst);
        __ RecordComment("]");
        __ RecordComment("[  Move(dst, temp);");
        __ Move(dst, temp);
        __ RecordComment("]");
      } else {
        DCHECK(destination->IsFPStackSlot());
        MemOperand dst = g.ToMemOperand(destination);
        if (rep == MachineRepresentation::kFloat32) {
          __ RecordComment("[  MoveFloat(temp, src);");
          __ MoveFloat(temp, src);
          __ RecordComment("]");
          __ RecordComment("[  LoadFloat(src, dst);");
          __ LoadFloat(src, dst);
          __ RecordComment("]");
          __ RecordComment("[  StoreFloat(temp, dst);");
          __ StoreFloat(temp, dst);
          __ RecordComment("]");
        } else {
          DCHECK_EQ(rep, MachineRepresentation::kFloat64);
          __ RecordComment("[  MoveDouble(temp, src);");
          __ MoveDouble(temp, src);
          __ RecordComment("]");
          __ RecordComment("[  LoadDouble(src, dst);");
          __ LoadDouble(src, dst);
          __ RecordComment("]");
          __ RecordComment("[  StoreDouble(temp, dst);");
          __ StoreDouble(temp, dst);
          __ RecordComment("]");
        }
      }
    }
  } else if (source->IsFPStackSlot()) {
    DCHECK(destination->IsFPStackSlot());
    Register temp_0 = kScratchReg;
    MemOperand src0 = g.ToMemOperand(source);
    MemOperand src1(src0.rm(), src0.offset() + kIntSize);
    MemOperand dst0 = g.ToMemOperand(destination);
    MemOperand dst1(dst0.rm(), dst0.offset() + kIntSize);
    MachineRepresentation rep = LocationOperand::cast(source)->representation();
    if (rep == MachineRepresentation::kSimd128) {
      UNIMPLEMENTED();
    } else {
      FPURegister temp_1 = kScratchDoubleReg;
      if (rep == MachineRepresentation::kFloat32) {
        __ RecordComment(
            "[  LoadFloat(temp_1, dst0);  // Save destination in temp_1.");
        __ LoadFloat(temp_1, dst0);  // Save destination in temp_1.
        __ RecordComment("]");
        __ RecordComment(
            "[  Lw(temp_0, src0);  // Then use temp_0 to copy source to "
            "destination.");
        __ Lw(temp_0, src0);  // Then use temp_0 to copy source to destination.
        __ RecordComment("]");
        __ RecordComment("[  Sw(temp_0, dst0);");
        __ Sw(temp_0, dst0);
        __ RecordComment("]");
        __ RecordComment("[  StoreFloat(temp_1, src0);");
        __ StoreFloat(temp_1, src0);
        __ RecordComment("]");
      } else {
        DCHECK_EQ(rep, MachineRepresentation::kFloat64);
        __ RecordComment(
            "[  LoadDouble(temp_1, dst0);  // Save destination in temp_1.");
        __ LoadDouble(temp_1, dst0);  // Save destination in temp_1.
        __ RecordComment("]");
        __ RecordComment(
            "[  Lw(temp_0, src0);  // Then use temp_0 to copy source to "
            "destination.");
        __ Lw(temp_0, src0);  // Then use temp_0 to copy source to destination.
        __ RecordComment("]");
        __ RecordComment("[  Sw(temp_0, dst0);");
        __ Sw(temp_0, dst0);
        __ RecordComment("]");
        __ RecordComment("[  Lw(temp_0, src1);");
        __ Lw(temp_0, src1);
        __ RecordComment("]");
        __ RecordComment("[  Sw(temp_0, dst1);");
        __ Sw(temp_0, dst1);
        __ RecordComment("]");
        __ RecordComment("[  StoreDouble(temp_1, src0);");
        __ StoreDouble(temp_1, src0);
        __ RecordComment("]");
      }
    }
  } else {
    // No other combinations are possible.
    UNREACHABLE();
  }
}

void CodeGenerator::AssembleJumpTable(Label** targets, size_t target_count) {
  // On 64-bit RISC-V we emit the jump tables inline.
  UNREACHABLE();
}

#undef ASSEMBLE_ATOMIC_LOAD_INTEGER
#undef ASSEMBLE_ATOMIC_STORE_INTEGER
#undef ASSEMBLE_ATOMIC_BINOP
#undef ASSEMBLE_ATOMIC_BINOP_EXT
#undef ASSEMBLE_ATOMIC_EXCHANGE_INTEGER
#undef ASSEMBLE_ATOMIC_EXCHANGE_INTEGER_EXT
#undef ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER
#undef ASSEMBLE_ATOMIC_COMPARE_EXCHANGE_INTEGER_EXT
#undef ASSEMBLE_IEEE754_BINOP
#undef ASSEMBLE_IEEE754_UNOP

#undef TRACE_MSG
#undef TRACE_UNIMPL
#undef __

}  // namespace compiler
}  // namespace internal
}  // namespace v8
