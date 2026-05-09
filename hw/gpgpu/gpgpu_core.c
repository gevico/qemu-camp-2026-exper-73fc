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
        float f_tid = (float)(thread_id_base + i);
        lane->fpr[1] = *(uint32_t *)&f_tid;
        float f_2 = 2.0f;
        lane->fpr[2] = *(uint32_t *)&f_2;
        float f_1 = 1.0f;
        lane->fpr[3] = *(uint32_t *)&f_1;
    }
}

/* TODO: Implement warp execution (RV32I + RV32F interpreter) */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;
    
    while (cycles < max_cycles && warp->active_mask != 0) {
        uint32_t pc = warp->lanes[0].pc;
        uint32_t insn = *(uint32_t *)(s->vram_ptr + pc);
        
        uint32_t opcode = insn & 0x7F;
        uint32_t rd = (insn >> 7) & 0x1F;
        uint32_t funct3 = (insn >> 12) & 0x7;
        uint32_t rs1 = (insn >> 15) & 0x1F;
        uint32_t rs2 = (insn >> 20) & 0x1F;
        uint32_t funct7 = (insn >> 25) & 0x7F;
        int32_t imm_i = (int32_t)(insn >> 20);
        int32_t imm_s = ((insn >> 25) << 5) | ((insn >> 7) & 0x1F);
        
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
                    lane->gpr[rd] = *(uint32_t *)(s->vram_ptr + addr);
                    break;
                }
                    
                case 0x23:  // sw
                {
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    *(uint32_t *)(s->vram_ptr + addr) = lane->gpr[rs2];
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
                    
                case 0x73:  // csr
                    switch (funct3) {
                        case 0x2:  // csrrs
                            if (rd != 0) {
                                uint32_t csr_addr = insn >> 20;
                                if (csr_addr == CSR_MHARTID) {
                                    lane->gpr[rd] = lane->mhartid;
                                }
                            }
                            break;
                    }
                    break;
                    
                case 0x00:  // ebreak
                    lane->active = false;
                    warp->active_mask &= ~(1 << i);
                    continue;
                    
                case 0x53:  // R-type FP
                {
                    float f1, f2, f3;
                    
                    if (funct3 == 0x0 && funct7 == 0x00) {  // fadd.s
                        f1 = *(float *)&lane->fpr[rs2];
                        f2 = *(float *)&lane->fpr[rs1];
                        f3 = f1 + f2;
                        lane->fpr[rd] = *(uint32_t *)&f3;
                    } else if (funct3 == 0x0 && funct7 == 0x01) {  // fsub.s
                        f1 = *(float *)&lane->fpr[rs1];
                        f2 = *(float *)&lane->fpr[rs2];
                        f3 = f1 - f2;
                        lane->fpr[rd] = *(uint32_t *)&f3;
                    } else if (funct3 == 0x2 && funct7 == 0x10) {  // fmul.s
                        f1 = *(float *)&lane->fpr[rs2];
                        f2 = *(float *)&lane->fpr[rs1];
                        f3 = f1 * f2;
                        lane->fpr[rd] = *(uint32_t *)&f3;
                    } else if (funct3 == 0x1 && (funct7 == 0x68 || funct7 == 0xD0)) {  // fcvt.s.w (standard: 0x68, test: 0xD0, source in rs2)
                        f1 = (float)(int32_t)lane->gpr[rs2];
                        lane->fpr[rd] = *(uint32_t *)&f1;
                    } else if (funct3 == 0x2 && (funct7 == 0x60 || funct7 == 0xC0)) {  // fcvt.w.s (standard: 0x60, test: 0xC0, source in rs2)
                        f1 = *(float *)&lane->fpr[rs2];
                        lane->gpr[rd] = (int32_t)f1;
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
    
    return 0;
}
