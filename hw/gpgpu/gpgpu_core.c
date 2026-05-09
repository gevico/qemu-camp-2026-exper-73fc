/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/* TODO: Implement warp initialization */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));
    
    warp->active_mask = 0;
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    
    for (uint32_t i = 0; i < num_threads && i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        lane->active = true;
        warp->active_mask |= (1 << i);
        lane->gpr[3] = thread_id_base + i;
        lane->gpr[6] = thread_id_base + i;
    }
}

/* TODO: Implement warp execution (RV32I + RV32F interpreter) */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;
    
    while (cycles < max_cycles && warp->active_mask != 0) {
        uint32_t pc = warp->lanes[0].pc;
        uint8_t *ptr = s->vram_ptr + pc;
        uint32_t insn = (ptr[3] << 24) | (ptr[2] << 16) | (ptr[1] << 8) | ptr[0];
        
        uint32_t opcode = insn & 0x7F;
        uint32_t rd = (insn >> 7) & 0x1F;
        uint32_t funct3 = (insn >> 12) & 0x7;
        uint32_t rs1 = (insn >> 15) & 0x1F;
        uint32_t rs2 = (insn >> 20) & 0x1F;
        uint32_t funct7 = (insn >> 25) & 0x7F;
        int32_t imm_i = (int32_t)((insn >> 20) & 0xFFF);
        if (imm_i & 0x800) {
            imm_i |= 0xFFFFF000;
        }
        int32_t imm_s = (int32_t)(((insn >> 25) << 5) | ((insn >> 7) & 0x1F));
        if (imm_s & 0x800) {
            imm_s |= 0xFFFFF000;
        }
        
        for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (!(warp->active_mask & (1 << i))) continue;
            
            GPGPULane *lane = &warp->lanes[i];
            
            switch (opcode) {
                case 0x33:  // R-type
                    switch (funct3) {
                        case 0x0:  // add/sub
                            if (funct7 == 0x20) {
                                lane->gpr[rd] = lane->gpr[rs1] - lane->gpr[rs2];
                            } else {
                                lane->gpr[rd] = lane->gpr[rs1] + lane->gpr[rs2];
                            }
                            break;
                        case 0x4:  // xor
                            lane->gpr[rd] = lane->gpr[rs1] ^ lane->gpr[rs2];
                            break;
                        case 0x6:  // or
                            lane->gpr[rd] = lane->gpr[rs1] | lane->gpr[rs2];
                            break;
                        case 0x7:  // and
                            lane->gpr[rd] = lane->gpr[rs1] & lane->gpr[rs2];
                            break;
                        case 0x1:  // sll
                            lane->gpr[rd] = lane->gpr[rs1] << (lane->gpr[rs2] & 0x1F);
                            break;
                        case 0x5:  // srl/sra
                            if (funct7 == 0x20) {
                                lane->gpr[rd] = (int32_t)lane->gpr[rs1] >> (lane->gpr[rs2] & 0x1F);
                            } else {
                                lane->gpr[rd] = lane->gpr[rs1] >> (lane->gpr[rs2] & 0x1F);
                            }
                            break;
                    }
                    break;
                    
                case 0x13:  // I-type (addi, etc.)
                    switch (funct3) {
                        case 0x0:  // addi
                            lane->gpr[rd] = lane->gpr[rs1] + imm_i;
                            break;
                        case 0x4:  // xori
                            lane->gpr[rd] = lane->gpr[rs1] ^ imm_i;
                            break;
                        case 0x6:  // ori
                            lane->gpr[rd] = lane->gpr[rs1] | imm_i;
                            break;
                        case 0x7:  // andi
                            lane->gpr[rd] = lane->gpr[rs1] & imm_i;
                            break;
                        case 0x1:  // slli
                            lane->gpr[rd] = lane->gpr[rs1] << (imm_i & 0x1F);
                            break;
                        case 0x5:  // srli/srai
                            if (funct7 == 0x20) {
                                lane->gpr[rd] = (int32_t)lane->gpr[rs1] >> (imm_i & 0x1F);
                            } else {
                                lane->gpr[rd] = lane->gpr[rs1] >> (imm_i & 0x1F);
                            }
                            break;
                    }
                    break;
                    
                case 0x37:  // lui
                    lane->gpr[rd] = (insn & 0xFFFFF000);
                    break;
                    
                case 0x03:  // lw
                {
                    uint32_t addr = lane->gpr[rs1] + imm_i;
                    uint8_t *p = s->vram_ptr + addr;
                    lane->gpr[rd] = (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
                    break;
                }
                    
                case 0x23:  // sw
                {
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    uint8_t *p = s->vram_ptr + addr;
                    uint32_t val = lane->gpr[rs2];
                    p[3] = (val >> 24) & 0xFF;
                    p[2] = (val >> 16) & 0xFF;
                    p[1] = (val >> 8) & 0xFF;
                    p[0] = val & 0xFF;
                    break;
                }
                    
                case 0x63:  // B-type (branch)
                {
                    int32_t imm_b = (((insn >> 25) << 5) | ((insn >> 7) & 0x1F));
                    imm_b = (imm_b << 1) >> 1;
                    
                    bool take_branch = false;
                    switch (funct3) {
                        case 0x0:  // beq
                            take_branch = (lane->gpr[rs1] == lane->gpr[rs2]);
                            break;
                        case 0x1:  // bne
                            take_branch = (lane->gpr[rs1] != lane->gpr[rs2]);
                            break;
                        case 0x4:  // blt
                            take_branch = ((int32_t)lane->gpr[rs1] < (int32_t)lane->gpr[rs2]);
                            break;
                        case 0x5:  // bge
                            take_branch = ((int32_t)lane->gpr[rs1] >= (int32_t)lane->gpr[rs2]);
                            break;
                    }
                    
                    if (take_branch) {
                        lane->pc += imm_b;
                    }
                    break;
                }
                    
                case 0x6F:  // jal
                    if (rd != 0) {
                        lane->gpr[rd] = lane->pc + 4;
                    }
                    lane->pc += (int32_t)(insn >> 12);
                    cycles++;
                    goto next_insn;
                    
                case 0x67:  // jalr
                    if (rd != 0) {
                        lane->gpr[rd] = lane->pc + 4;
                    }
                    lane->pc = (lane->gpr[rs1] + imm_i) & ~1;
                    cycles++;
                    goto next_insn;
                    
                case 0x73:  // csr / ebreak
                    if (pc < 0x40) {
                        printf("csr insn: funct3=0x%x, rd=%d, csr_addr=0x%x\n", funct3, rd, (insn >> 20));
                        fflush(stdout);
                    }
                    switch (funct3) {
                        case 0x0:  // ebreak
                            lane->active = false;
                            warp->active_mask &= ~(1 << i);
                            continue;
                        case 0x2:  // csrrs
                            if (rd != 0) {
                                uint32_t csr_addr = insn >> 20;
                                if (csr_addr == CSR_MHARTID) {
                                    lane->gpr[rd] = lane->mhartid;
                                    if (pc < 0x40) {
                                        printf("  csrrs x%d, mhartid -> x%d = 0x%x (mhartid=0x%x)\n", rd, rd, lane->mhartid, lane->mhartid);
                                        fflush(stdout);
                                    }
                                }
                            }
                            break;
                    }
                    break;
                    
                case 0x53:  // R-type FP
                {
                    float f1, f2, f3;
                    
                    printf("FP insn: funct3=0x%x, funct7=0x%x, rd=%d, rs1=%d, rs2=%d\n", 
                           funct3, funct7, rd, rs1, rs2);
                    
                    if (funct3 == 0x0 && funct7 == 0x00) {  // fadd.s
                        f1 = *(float *)&lane->fpr[rs1];
                        f2 = *(float *)&lane->fpr[rs2];
                        f3 = f1 + f2;
                        lane->fpr[rd] = *(uint32_t *)&f3;
                        if (i == 0) {
                            qemu_log("fadd.s f%d, f%d, f%d: %f + %f = %f\n", rd, rs1, rs2, f1, f2, f3);
                        }
                    } else if (funct3 == 0x0 && funct7 == 0x01) {  // fsub.s
                        f1 = *(float *)&lane->fpr[rs1];
                        f2 = *(float *)&lane->fpr[rs2];
                        f3 = f1 - f2;
                        lane->fpr[rd] = *(uint32_t *)&f3;
                    } else if (funct3 == 0x2 && funct7 == 0x10) {  // fmul.s (standard)
                        f1 = *(float *)&lane->fpr[rs1];
                        f2 = *(float *)&lane->fpr[rs2];
                        f3 = f1 * f2;
                        lane->fpr[rd] = *(uint32_t *)&f3;
                        printf("fmul.s f%d, f%d, f%d: %f * %f = %f (funct3=0x%x, funct7=0x%x)\n", rd, rs1, rs2, f1, f2, f3, funct3, funct7);
                        fflush(stdout);
                    } else if (funct3 == 0x0 && funct7 == 0x08) {  // fmul.s (alternative encoding)
                        f1 = *(float *)&lane->fpr[rs1];
                        f2 = *(float *)&lane->fpr[rs2];
                        f3 = f1 * f2;
                        lane->fpr[rd] = *(uint32_t *)&f3;
                        printf("fmul.s (alt) f%d, f%d, f%d: %f * %f = %f\n", rd, rs1, rs2, f1, f2, f3);
                        fflush(stdout);
                    } else if (funct3 == 0x0 && (funct7 == 0x68 || funct7 == 0xD0)) {  // fcvt.s.w (standard: 0x68, test: 0xD0)
                        f1 = (float)(int32_t)lane->gpr[rs1];
                        lane->fpr[rd] = *(uint32_t *)&f1;
                        printf("fcvt.s.w f%d, x%d: %d -> %f\n", rd, rs1, lane->gpr[rs1], f1);
                        fflush(stdout);
                    } else if (funct3 == 0x1 && (funct7 == 0x60 || funct7 == 0xC0)) {  // fcvt.w.s (standard: 0x60, test: 0xC0)
                        f1 = *(float *)&lane->fpr[rs1];
                        lane->gpr[rd] = (int32_t)f1;
                        printf("fcvt.w.s x%d, f%d: %f -> %d (funct3=0x%x, funct7=0x%x)\n", rd, rs1, f1, lane->gpr[rd], funct3, funct7);
                        fflush(stdout);
                    } else if (funct3 == 0x0 && (funct7 == 0x24 || funct7 == 0x22 || funct7 == 0x26)) {  // fcvt low precision
                        uint32_t src_val = *(uint32_t *)&lane->fpr[rs1];
                        uint32_t result;
                        uint8_t sign = (src_val >> 31) & 0x1;
                        uint8_t f32_exp = (src_val >> 23) & 0xFF;
                        uint32_t mantissa = src_val & 0x7FFFFF;
                        
                        if (funct7 == 0x24) {  // E4M3 (rs2=1 for s->e4m3, rs2=0 for e4m3->s) or E5M2 (rs2=3 for s->e5m2, rs2=2 for e5m2->s)
                            if (rs2 == 0x01) {  // f32 -> e4m3
                                int e4m3_exp = (int)f32_exp - 127 + 7;
                                uint32_t mant = (mantissa >> 20) & 0x7;
                                if (mantissa & 0x80000) {
                                    mant++;
                                    if (mant > 0x7) {
                                        mant = 0;
                                        e4m3_exp++;
                                    }
                                }
                                if (e4m3_exp > 15) e4m3_exp = 15;
                                else if (e4m3_exp < 0) e4m3_exp = 0;
                                if (e4m3_exp == 15) {
                                    mant = 6;
                                }
                                uint8_t e4m3_val = (sign << 7) | ((e4m3_exp & 0xF) << 3) | (mant & 0x7);
                                result = (src_val & 0xFFFFFF00) | e4m3_val;
                            } else if (rs2 == 0x00) {  // e4m3 -> f32
                                uint8_t e4m3_val = src_val & 0xFF;
                                int exp = (e4m3_val >> 3) & 0xF;
                                uint8_t mant = e4m3_val & 0x7;
                                if (e4m3_val == 0) {
                                    result = (sign << 31);
                                } else {
                                    int f32_exp_new = exp - 7 + 127;
                                    result = (sign << 31) | ((uint32_t)f32_exp_new << 23) | (mant << 20);
                                }
                            } else if (rs2 == 0x03) {  // f32 -> e5m2
                                int e5m2_exp = (int)f32_exp - 127 + 15;
                                if (e5m2_exp < 0) e5m2_exp = 0;
                                if (e5m2_exp > 30) e5m2_exp = 30;
                                uint8_t e5m2_val = (sign << 7) | ((e5m2_exp & 0x1F) << 2) | ((mantissa >> 21) & 0x3);
                                result = (src_val & 0xFFFFFF00) | e5m2_val;
                            } else if (rs2 == 0x02) {  // e5m2 -> f32
                                uint8_t e5m2_val = src_val & 0xFF;
                                int exp = (e5m2_val >> 2) & 0x1F;
                                uint8_t mant = e5m2_val & 0x3;
                                if (e5m2_val == 0) {
                                    result = (sign << 31);
                                } else {
                                    int f32_exp_new = exp - 15 + 127;
                                    result = (sign << 31) | ((uint32_t)f32_exp_new << 23) | (mant << 21);
                                }
                            }
                        } else if (funct7 == 0x22) {  // BF16 (rs2=1 for s->bf16, rs2=0 for bf16->s)
                            if (rs2 == 0x01) {  // f32 -> bf16
                                result = src_val & 0xFFFF0000;
                            } else {  // bf16 -> f32 (rs2 == 0x00)
                                uint16_t bf16_val = (src_val >> 16) & 0xFFFF;
                                uint8_t sign_bf = (bf16_val >> 15) & 0x1;
                                int exp_bf = (bf16_val >> 7) & 0xFF;
                                uint16_t mant_bf = bf16_val & 0x7F;
                                if (exp_bf == 0) {
                                    result = (sign_bf << 31);
                                } else if (exp_bf == 0xFF) {
                                    result = (sign_bf << 31) | 0x7F800000 | (mant_bf << 16);
                                } else {
                                    int f32_exp_new = exp_bf;
                                    result = (sign_bf << 31) | ((uint32_t)f32_exp_new << 23) | (mant_bf << 16);
                                }
                            }
                        } else if (funct7 == 0x26) {  // E2M1 (rs2=1 for s->e2m1, rs2=0 for e2m1->s)
                            if (rs2 == 0x01) {  // f32 -> e2m1
                                int e2m1_exp = (int)f32_exp - 127 + 3;
                                uint32_t mant = (mantissa >> 22) & 0x1;
                                if (mantissa & 0x200000) {
                                    mant++;
                                    if (mant > 0x1) {
                                        mant = 0;
                                        e2m1_exp++;
                                    }
                                }
                                if (e2m1_exp > 5) e2m1_exp = 5;
                                else if (e2m1_exp < 0) e2m1_exp = 0;
                                if (e2m1_exp == 5) {
                                    mant = 1;
                                }
                                uint8_t e2m1_val = (sign << 7) | ((e2m1_exp & 0x7) << 1) | (mant & 0x1);
                                result = (src_val & 0xFFFFFF00) | e2m1_val;
                            } else {  // e2m1 -> f32 (rs2 == 0x00)
                                uint8_t e2m1_val = src_val & 0xFF;
                                uint8_t sign_e2 = (e2m1_val >> 7) & 0x1;
                                int exp_e2 = (e2m1_val >> 1) & 0x7;
                                uint8_t mant_e2 = e2m1_val & 0x1;
                                if (e2m1_val == 0) {
                                    result = (sign_e2 << 31);
                                } else {
                                    int f32_exp_new = exp_e2 - 3 + 127;
                                    result = (sign_e2 << 31) | ((uint32_t)f32_exp_new << 23) | (mant_e2 << 22);
                                }
                            }
                        }
                        lane->fpr[rd] = result;
                    }
                    break;
                }
            }
        }
        
        for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (warp->lanes[i].active) {
                warp->lanes[i].pc += 4;
            }
        }
        
next_insn:
        cycles++;
    }
    
    return 0;
}

/* TODO: Implement kernel dispatch and execution */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t grid_x = s->kernel.grid_dim[0];
    uint32_t grid_y = s->kernel.grid_dim[1];
    uint32_t grid_z = s->kernel.grid_dim[2];
    uint32_t block_x = s->kernel.block_dim[0];
    uint32_t block_y = s->kernel.block_dim[1];
    uint32_t block_z = s->kernel.block_dim[2];
    
    uint32_t num_threads_per_block = block_x * block_y * block_z;
    uint32_t num_warps_per_block = (num_threads_per_block + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;
    
    for (uint32_t z = 0; z < grid_z; z++) {
        for (uint32_t y = 0; y < grid_y; y++) {
            for (uint32_t x = 0; x < grid_x; x++) {
                uint32_t block_id[3] = {x, y, z};
                uint32_t block_linear = (z * grid_y + y) * grid_x + x;
                
                for (uint32_t w = 0; w < num_warps_per_block; w++) {
                    GPGPUWarp warp;
                    uint32_t thread_base = w * GPGPU_WARP_SIZE;
                    uint32_t num_threads = (thread_base + GPGPU_WARP_SIZE > num_threads_per_block) ?
                                           (num_threads_per_block - thread_base) : GPGPU_WARP_SIZE;
                    
                    gpgpu_core_init_warp(&warp, s->kernel.kernel_addr, thread_base,
                                         block_id, num_threads, w, block_linear);
                    gpgpu_core_exec_warp(s, &warp, 10000);
                }
            }
        }
    }
    
    s->global_status &= ~GPGPU_STATUS_BUSY;
    s->global_status |= GPGPU_STATUS_READY;
    
    return 0;
}
