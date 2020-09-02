	.text
	;.amdgcn_target "amdgcn-amd-amdhsa--gfx908+sram-ecc"
	.weak	gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw ; -- Begin function gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw
	.p2align	8
	.type	gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw,@function
gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw: ; @gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw
; %bb.0:                                ; %entry



.set sgpr_ker_arg,4;
.set sgpr_workgroup_id,6;

;.set sgpr_C,6;
;.set sgpr_R,7;
.set sgpr_N,8;
.set sgpr_C,9;
.set sgpr_C0,10;
.set sgpr_Hi,11;
.set sgpr_Wi,12;
.set sgpr_datatype_log2,13;//float:2 fp16:1
.set sgpr_load_everytime,14
.set sgpr_threads,15

.set sgpr_HiWi,33
.set sgpr_CHiWi,35

.set sgpr_kevin_test_float_addr,16;
.set sgpr_kevin_test_uint_addr,18;
.set sgpr_A_addr,20;
.set sgpr_B_addr,22;
;.set sgpr_Ho,36;
;.set sgpr_Wo,37;

.set sgpr_C_addr,34;
.set sgpr_buf_read_addr,24;//24 25 26 27
.set sgpr_before_cmp_thread,28;//28 29
.set sgpr_tmp_cmp_thread,30;//30 31
.set Srd127_96, 0x0020000
.set S32,32
.set sgpr_32,32
;.set sgpr_CHiWi,33
.set sgpr_base_hwid_every_round, 33
.set sgpr_base_cid_every_round, 34


.set sgpr_threads_package_size,35
.set sgpr_threads_package_size_log2,46


.set sgpr_move_bytes,36
.set sgpr_CHiWi,37
;.set sgpr_write_c_1_threads,37
.set sgpr_block_start_addr,38;//38 39
.set sgpr_buf_read_addr,40;//40 41 42 43
.set sgpr_buf_write_addr,40;//40 41 42 43
.set sgpr_read_limit,44
.set sgpr_loop_num,45
.set sgpr_tmp_int,46
.set sgpr_div_tmp4,48;// 48 49 50 51
.set sgpr_read_limit_n,48

.set sgpr_tmp_cmp_positive,52;//52 53


.set vgpr_thread_id,0
.set vgpr_store_addr,2;//2 3


.set vgpr_thread_A_addr,4;//4 5 load

.set vgpr_tmp_sgpr,6
;.set vgpr_tmp_c0_tid,7 

.set vgpr_read_value,8;//8 9 10 11
.set vgpr_lds_addr_1,12
.set vgpr_lds_addr_2,13
.set vgpr_lds_addr_3,14
.set vgpr_lds_addr_4,15
.set vgpr_lds_addr_5,16
.set vgpr_lds_addr_6,17
.set vgpr_lds_addr_7,18
.set vgpr_lds_addr_8,19
.set vgpr_interval,20
.set vgpr_lds_read_offset,21


.set vgpr_lds_read8,24;24 25 26 27

.set vgpr_div_tmp4,28;//28 29 30 31
.set vgpr_tmp_1,29
.set vgpr_write_c_per_thread,32
.set vgpr_write_hw_id,33
.set vgpr_write_c_id,34

.set vgpr_thread_write_offset,36;//36 37


;.set vgpr_tmp_int,9
;.set vgpr_wave_tid,10
;.set vgpr_wave_tid_offset_8,11
;.set vgpr_wave_tid_offset_2,12
;.set vgpr_A_value,18;//18 19
;.set vgpr_wave_tid_offset_4,13
;.set vgpr_tmp_write_pos,124

;.set vgpr_A_mfma_value1,14;14:15
;.set vgpr_B_mfma_value1,16;16:!7

;.set vgpr_B_tid,8
;.set vgpr_tmp,7

.set vgpr_B_ushort1,29
;.set vgpr_B_ushort2,21
;.set vgpr_B_ushort3,22
;.set vgpr_B_ushort4,23


;.set vgpr_B_tid_div_4,24
;.set vgpr_B_tid_rem_4,25
;.set vgpr_B_read_offset_first,26;//26 27

;.set vgpr_B_tid_offset_2,28;//28  




.macro decide_threads_package_log2 s_threads, s_load_vectors, s_packages, s_packages_log2;//we use s_load_vectors =8 here, should add more after
s_or_saveexec_b64 s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1],exec
s_cmp_eq_u32  s[\s_threads], 64

s_cbranch_scc0 not_64
s_mov_b32 s[\s_packages],8
s_mov_b32 s[\s_packages_log2],3
s_branch decide_end

not_64:
s_cmp_eq_u32  s[\s_threads], 128
s_cbranch_scc0 not_64_128
s_mov_b32 s[\s_packages],16
s_mov_b32 s[\s_packages_log2],4
s_branch decide_end

not_64_128:
s_mov_b32 s[\s_packages],32
s_mov_b32 s[\s_packages_log2],5

decide_end:
s_mov_b64 exec,s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1]
.endm



.macro div_int_vv_rm v_r,v_q, v_n, v_d, v_tmp4, s_tmp4;v_q = v_n / v_d, v_r = v_n % v_d
v_cvt_f32_u32     v[\v_tmp4+0],   v[\v_d]
v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
v_mul_lo_u32      v[\v_tmp4+1],   v[\v_d],      v[\v_tmp4+0]
v_mul_hi_u32      v[\v_tmp4+2],   v[\v_d],      v[\v_tmp4+0]
v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
v_mul_hi_u32      v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_n]
v_mul_lo_u32      v[\v_tmp4+1],   v[\v_tmp4+0],   v[\v_d]
v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_n],      v[\v_tmp4+1]
v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], v[\v_n],      v[\v_tmp4+1]
v_cmp_ge_u32      s[\s_tmp4+2:\s_tmp4+3], v[\v_tmp4+2],   v[\v_d]
v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
v_cmp_ne_i32      vcc,          0,          v[\v_d]
v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
v_mul_lo_u32 v[\v_tmp4], v[\v_d], v[\v_q]
v_sub_u32 v[\v_r], v[\v_n], v[\v_tmp4]
.endm









s_load_dwordx2 s[sgpr_A_addr:sgpr_A_addr+1], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x0
s_load_dwordx2 s[sgpr_B_addr:sgpr_B_addr+1], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x8
s_load_dwordx2 s[sgpr_C_addr:sgpr_C_addr+1], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x10
;s_load_dword s[sgpr_C], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x18

s_load_dwordx2 s[sgpr_kevin_test_uint_addr:sgpr_kevin_test_uint_addr+1], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x18
s_load_dwordx2 s[sgpr_kevin_test_float_addr:sgpr_kevin_test_float_addr+1], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x20


s_load_dwordx4 s[sgpr_N:sgpr_Hi], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x28

s_load_dwordx2 s[sgpr_Wi:sgpr_datatype_log2], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x38
s_load_dwordx2 s[sgpr_load_everytime:sgpr_threads], s[sgpr_ker_arg:sgpr_ker_arg+1], 0x40

s_mov_b32 s[sgpr_base_hwid_every_round],0
s_mov_b32 s[sgpr_base_cid_every_round],0



decide_threads_package_log2 sgpr_threads, sgpr_load_everytime, sgpr_threads_package_size, sgpr_threads_package_size_log2

;//hw_id=(tid/sgpr_threads_package_size)
;//c_id =(tid%sgpr_threads_package_size)
v_lshrrev_b32_e32 v[vgpr_write_hw_id] ,s[sgpr_threads_package_size_log2],v[vgpr_thread_id]
v_mov_b32_e32 v[vgpr_div_tmp4],s[sgpr_threads_package_size_log2]
v_mov_b32_e32 v[vgpr_tmp_1],1
v_lshlrev_b32_e32 v[vgpr_div_tmp4+1],v[vgpr_div_tmp4],v[vgpr_tmp_1]
v_mul_lo_u32 v[vgpr_div_tmp4+1],v[vgpr_div_tmp4+1],v[vgpr_write_hw_id]
v_sub_u32_e32 v[vgpr_write_c_id], v[vgpr_thread_id], v[vgpr_div_tmp4+1]

;//final writre addr = (hwid*chw+cid)*2, but everytime hw should +8,c should +threads
;v_lshrrev_b32_e32 v[vgpr_write_c_per_thread],3,v[vgpr_thread_id]
;//calculate 
;div_int_vv_rm vgpr_write_c_8_id,vgpr_write_hw_id,vgpr_thread_id,vgpr_write_c_per_thread,vgpr_div_tmp4,sgpr_div_tmp4





s_waitcnt     lgkmcnt(0);

s_mul_i32 s[sgpr_HiWi],s[sgpr_Wi],s[sgpr_Hi]
s_mul_i32 s[sgpr_CHiWi],s[sgpr_C],s[sgpr_HiWi]


v_mul_lo_u32 v[vgpr_thread_write_offset],v[vgpr_write_hw_id],s[sgpr_CHiWi]
v_add_u32_e32 v[vgpr_thread_write_offset],v[vgpr_thread_write_offset],v[vgpr_write_c_id]
v_lshlrev_b32_e32 v[vgpr_thread_write_offset],s[sgpr_datatype_log2],v[vgpr_thread_write_offset]

s_mov_b32 s[sgpr_read_limit],s[sgpr_HiWi]
s_mul_i32 s[sgpr_read_limit],s[sgpr_read_limit],s[sgpr_C]
s_add_u32 s[sgpr_read_limit_n],s[sgpr_workgroup_id],1
s_mul_i32 s[sgpr_read_limit],s[sgpr_read_limit], s[sgpr_read_limit_n]
s_lshl_b32 s[sgpr_read_limit],s[sgpr_read_limit],s[sgpr_datatype_log2];//(n+1) * Chw*2

s_lshl_b32  s[sgpr_block_start_addr],s[sgpr_CHiWi],s[sgpr_datatype_log2];//bytes
s_mul_i32 s[sgpr_block_start_addr],s[sgpr_workgroup_id],s[sgpr_block_start_addr];n*CHW*2
s_mov_b32 s[sgpr_block_start_addr+1],0



;s_add_u32     s[sgpr_block_start_addr], s[sgpr_B_addr], s[sgpr_block_start_addr]
;s_addc_u32    s[sgpr_block_start_addr+1], s[sgpr_B_addr+1], 0


v_mul_lo_u32 v[vgpr_thread_A_addr],v[vgpr_thread_id],s[sgpr_HiWi];tid*hw
v_lshlrev_b32_e32 v[vgpr_thread_A_addr], s[sgpr_datatype_log2], v[vgpr_thread_A_addr];//tid*HW*2; even you have c ,everytime every thread handle 1 c





;s_mov_b32 s[sgpr_buf_read_addr],s[sgpr_block_start_addr]
;s_mov_b32 s[sgpr_buf_read_addr+1],s[sgpr_block_start_addr+1]
s_add_u32 s[sgpr_buf_read_addr],s[sgpr_B_addr], s[sgpr_block_start_addr]
s_addc_u32 s[sgpr_buf_read_addr+1],s[sgpr_B_addr+1], s[sgpr_block_start_addr+1]

s_mov_b32 s[sgpr_buf_read_addr+2],-1
s_mov_b32 s[sgpr_buf_read_addr+3],0x27000;s[sgpr_read_limit]

;v_mov_b32_e32 v[vgpr_thread_A_addr],0
buffer_load_dwordx4 v[vgpr_read_value:vgpr_read_value+3],v[vgpr_thread_A_addr],s[sgpr_buf_read_addr:sgpr_buf_read_addr+3],0, offen offset:0
;buffer_load_ushort v[vgpr_read_value],v[vgpr_thread_A_addr],s[sgpr_buf_read_addr:sgpr_buf_read_addr+3],0, offen offset:0

s_waitcnt vmcnt(0)
;t0->0 n 0 00
;t1->2 n 1 00
;t2->4 n 2 00
;t3->6 n 3 00  if we want to write n 00 c,easy
;we read threads*8(nchw 8 along hw direction)
;we write 8(nhwc along c)*threads
ds_write_b16_d16_hi v[vgpr_lds_addr_1],v[vgpr_read_value] offset:0 ;//hw0 
ds_write_b16 v[vgpr_lds_addr_2],v[vgpr_read_value] offset:0        ;hw1
ds_write_b16_d16_hi v[vgpr_lds_addr_3],v[vgpr_read_value+1] offset:0
ds_write_b16 v[vgpr_lds_addr_4],v[vgpr_read_value+1] offset:0

ds_write_b16_d16_hi v[vgpr_lds_addr_5],v[vgpr_read_value+2] offset:0
ds_write_b16 v[vgpr_lds_addr_6],v[vgpr_read_value+2] offset:0
ds_write_b16_d16_hi v[vgpr_lds_addr_7],v[vgpr_read_value+3] offset:0
ds_write_b16 v[vgpr_lds_addr_8],v[vgpr_read_value+3] offset:0

www:
s_add_u32 s[sgpr_buf_write_addr],s[sgpr_kevin_test_float_addr], s[sgpr_block_start_addr]
s_addc_u32 s[sgpr_buf_write_addr+1],s[sgpr_kevin_test_float_addr+1], s[sgpr_block_start_addr+1]
s_mov_b32 s[sgpr_buf_write_addr+2],-1
s_mov_b32 s[sgpr_buf_write_addr+3],0x27000;




kevin_write_test:


s_waitcnt     lgkmcnt(0)
s_waitcnt vmcnt(0)

s_or_saveexec_b64 s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1],exec
s_cmp_eq_u32  s[sgpr_workgroup_id], 0
s_cbranch_scc0 read_end


s_mov_b32 s63,35
v_cmpx_eq_u32 s[sgpr_tmp_cmp_positive:sgpr_tmp_cmp_positive+1], v[vgpr_thread_id],s63


;v_mov_b32_e32 v[vgpr_B_ushort1],v[vgpr_read_value]
;v_cvt_f16_f32_e32     v[vgpr_B_ushort1],v[vgpr_B_ushort1]

v_mov_b32_e32 v[vgpr_store_addr],0
v_mov_b32_e32 v[vgpr_store_addr+1],0

buffer_store_dwordx4 v[vgpr_read_value:vgpr_read_value+3],v[vgpr_thread_A_addr],s[sgpr_buf_write_addr:sgpr_buf_write_addr+3],0, offen offset:0

v_mov_b32_e32 v[vgpr_write_c_id], s[sgpr_CHiWi]
global_store_dword v[vgpr_store_addr:vgpr_store_addr+1], v[vgpr_write_c_id], s[sgpr_kevin_test_uint_addr:sgpr_kevin_test_uint_addr+1]
;global_store_short_d16_hi v[vgpr_store_addr:vgpr_store_addr+1], v[vgpr_read_value], s[sgpr_kevin_test_float_addr:sgpr_kevin_test_float_addr+1]



s_waitcnt vmcnt(0)


s_nor_saveexec_b64 exec,s[sgpr_tmp_cmp_positive:sgpr_tmp_cmp_positive+1] ;//not thread 0



;v_lshrrev_b16 



;V_LSHRREV_B16

;v_cvt_f16_f32_e32     v[vgpr_B_ushort1],v[vgpr_B_ushort1]





/*
s_lshl_b32 s[sgpr_move_bytes],2,s[sgpr_datatype_log2]

s_mul_i32 s[sgpr_HiWi],s[sgpr_Wi],s[sgpr_Hi]
s_mul_i32 s[sgpr_CHiWi],s[sgpr_C],s[sgpr_HiWi]



;s_mul_i32 s[sgpr_read_limit],s[sgpr_C0],s[sgpr_HiWi]
;s_mul_i32 s[sgpr_read_limit],s[sgpr_read_limit],s[sgpr_C]
s_mov_b32 s[sgpr_read_limit],s[sgpr_HiWi]
s_lshl_b32 s[sgpr_read_limit],s[sgpr_read_limit],s[sgpr_datatype_log2];//hw*2

s_lshl_b32  s[sgpr_block_start_addr],s[sgpr_CHiWi],s[sgpr_datatype_log2];//bytes
s_mul_i32 s[sgpr_block_start_addr],s[sgpr_workgroup_id],s[sgpr_block_start_addr];n*CHW*2
s_mov_b32 s[sgpr_block_start_addr+1],0




v_mul_lo_u32 v[vgpr_thread_A_addr],v[vgpr_thread_id],s[sgpr_HiWi];tid*hw


v_lshlrev_b32_e32 v[vgpr_thread_A_addr], s[sgpr_datatype_log2], v[vgpr_thread_A_addr];//tid*HW*2
v_mov_b32_e32 v[vgpr_thread_A_addr+1],0



;you need a loop to handle 0 to c0,everytime add hw*2


s_mov_b32 s[sgpr_loop_num],0

read_start_everytime_8:


s_mov_b32 s[sgpr_buf_A_addr],s[sgpr_block_start_addr]
s_mov_b32 s[sgpr_buf_A_addr+1],s[sgpr_block_start_addr+1]
s_mov_b32 s[sgpr_buf_A_addr+2],-1
s_mov_b32 s[sgpr_buf_A_addr+3],s[sgpr_read_limit]
buffer_load_dwordx4 v[vgpr_A_value:vgpr_A_value+3],v[vgpr_thread_A_addr],s[sgpr_buf_A_addr:sgpr_buf_A_addr+3],0, offen offset:0


;s_mul_i32 s[],
v_lshlrev_b32_e32 v[vgpr_lds_addr_1],s[sgpr_datatype_log2],v[vgpr_thread_id]
v_mov_b32_e32 v[vgpr_tmp_sgpr],s[sgpr_threads]
v_lshlrev_b32_e32 v[vgpr_interval],s[sgpr_datatype_log2],v[vgpr_tmp_sgpr]
v_add_u32_e32 v[vgpr_lds_addr_2], v[vgpr_lds_addr_1],v[vgpr_interval]
v_add_u32_e32 v[vgpr_lds_addr_3], v[vgpr_lds_addr_2],v[vgpr_interval]
v_add_u32_e32 v[vgpr_lds_addr_4], v[vgpr_lds_addr_3],v[vgpr_interval]
v_add_u32_e32 v[vgpr_lds_addr_5], v[vgpr_lds_addr_4],v[vgpr_interval]
v_add_u32_e32 v[vgpr_lds_addr_6], v[vgpr_lds_addr_5],v[vgpr_interval]
v_add_u32_e32 v[vgpr_lds_addr_7], v[vgpr_lds_addr_6],v[vgpr_interval]
v_add_u32_e32 v[vgpr_lds_addr_8], v[vgpr_lds_addr_7],v[vgpr_interval]

;v_mul_lo_u32 v[vgpr_lds_addr_1],v[vgpr_thread_id],
;vgpr_lds_addr_1
s_waitcnt vmcnt(0)
ds_write_b16_d16_hi v[vgpr_lds_addr_1],v[vgpr_A_value] offset:0 ;//hw0 
ds_write_b16 v[vgpr_lds_addr_2],v[vgpr_A_value] offset:0        ;hw1
ds_write_b16_d16_hi v[vgpr_lds_addr_3],v[vgpr_A_value+1] offset:0
ds_write_b16 v[vgpr_lds_addr_4],v[vgpr_A_value+1] offset:0

ds_write_b16_d16_hi v[vgpr_lds_addr_5],v[vgpr_A_value+2] offset:0
ds_write_b16 v[vgpr_lds_addr_6],v[vgpr_A_value+2] offset:0
ds_write_b16_d16_hi v[vgpr_lds_addr_7],v[vgpr_A_value+3] offset:0
ds_write_b16 v[vgpr_lds_addr_8],v[vgpr_A_value+3] offset:0


lds_write_8:

;s_lshr_b32 s[sgpr_write_c_1_threads],s[sgpr_threads],3

;//vgpr_lds_read_offset:8*tid*2
v_lshlrev_b32_e32 v[vgpr_lds_read_offset], s[sgpr_datatype_log2], v[vgpr_thread_id];
v_lshlrev_b32_e32 v[vgpr_lds_read_offset],3, v[vgpr_lds_read_offset]


s_waitcnt     lgkmcnt(0);
ds_read_b128 v[vgpr_lds_read8:vgpr_lds_read8+3], v[vgpr_lds_read_offset] offset:0




write_to_global:
v_mul_lo_u32 v[vgpr_thread_A_write_offset],v[vgpr_write_hw_id],s[sgpr_C];
v_lshlrev_b32_e32 v[vgpr_thread_A_write_offset],s[sgpr_datatype_log2],v[vgpr_thread_A_write_offset];//hw*c*2
v_lshlrev_b32_e32 v[vgpr_div_tmp4],3,v[vgpr_write_c_8_id];//tmp vgpr overwrite
v_lshlrev_b32_e32 v[vgpr_div_tmp4],s[sgpr_datatype_log2],v[vgpr_div_tmp4];//c_8*2
v_add_u32_e32 v[vgpr_thread_A_write_offset],v[vgpr_thread_A_write_offset],v[vgpr_div_tmp4];//hw*c*2+c_8*2,if add n*HWC*2 ,it's write addr


s_waitcnt     lgkmcnt(0);




;//here is test
s_or_saveexec_b64 s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1],exec
s_cmp_eq_u32  s[sgpr_workgroup_id], 3
s_cbranch_scc0 read_end


write_test:

v_mov_b32_e32 v[vgpr_B_ushort1],s[sgpr_block_start_addr]
v_mov_b32_e32 v[vgpr_store_addr],0
v_mov_b32_e32 v[vgpr_store_addr+1],0
global_store_dword v[vgpr_store_addr:vgpr_store_addr+1], v[vgpr_B_ushort1], s[sgpr_kevin_test_float_addr:sgpr_kevin_test_float_addr+1]
s_waitcnt vmcnt(0)

*/



;//


/*
s_mov_b32 s[sgpr_32],sgpr_32

v_and_b32_e32     v[vgpr_wave_tid], 63, v0;
s_or_saveexec_b64 s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1],exec
v_cmpx_lt_u32 s[sgpr_tmp_cmp_thread:sgpr_tmp_cmp_thread+1], v0, s[sgpr_32]
;s_cbranch_scc0 read_B

read_A:

//v_mov_b32_e32 v2,s[sgpr_C]
//v_mov_b32_e32 v1,s[sgpr_kevin_test_uint_addr+1]
//v_mov_b32_e32 v0,s[sgpr_kevin_test_uint_addr]
//global_store_dword v[0:1], v2, off offset:0


v_lshlrev_b32_e32 v[vgpr_wave_tid_offset_8], 3, v[vgpr_wave_tid];//every thread read 4 fp16, address: 4*2*tid
v_lshlrev_b32_e32 v[vgpr_wave_tid_offset_4], 2, v[vgpr_wave_tid]
v_lshlrev_b32_e32 v[vgpr_wave_tid_offset_2], 1, v[vgpr_wave_tid]


v_mov_b32_e32 v[vgpr_store_addr],v[vgpr_wave_tid_offset_4]
v_mov_b32_e32 v[vgpr_store_addr+1],0
v_mov_b32_e32 v[vgpr_tmp_int],2
global_store_dword v[vgpr_store_addr:vgpr_store_addr+1], v0, s[sgpr_kevin_test_uint_addr:sgpr_kevin_test_uint_addr+1]

s_waitcnt vmcnt(0)


;here debug
;v_mov_b32_e32 v[vgpr_wave_tid_offset],0
s_mov_b32 s[sgpr_buf_A_addr],s[sgpr_A_addr]
s_mov_b32 s[sgpr_buf_A_addr+1],s[sgpr_A_addr+1]
s_mov_b32 s[sgpr_buf_A_addr+2],-1
s_mov_b32 s[sgpr_buf_A_addr+3],0x40000;Srd127_96

buffer_load_dwordx2 v[vgpr_A_value:vgpr_A_value+1],v[vgpr_wave_tid_offset_8],s[sgpr_buf_A_addr:sgpr_buf_A_addr+3],0, offen offset:0





;v_mov_b32_e32 v[vgpr_wave_tid_offset+1],0
;global_load_ushort v[vgpr_A_value],v[vgpr_wave_tid_offset:vgpr_wave_tid_offset+1],s[sgpr_A_addr:sgpr_A_addr+1]
s_waitcnt vmcnt(0)

*/



/*a store
v_mov_b32_e32 v[vgpr_store_addr],v[vgpr_wave_tid_offset_2]
v_mov_b32_e32 v[vgpr_store_addr+1],0


global_store_short v[vgpr_store_addr:vgpr_store_addr+1], v[vgpr_A_value], s[sgpr_kevin_test_float_addr:sgpr_kevin_test_float_addr+1]
*/

;s_branch read_end


/*
read_B:

s_mov_b64 exec,s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1]

v_cmpx_ge_u32 s[sgpr_tmp_cmp_thread:sgpr_tmp_cmp_thread+1], v0, s[sgpr_32]

v_mov_b32_e32 v[vgpr_tmp],s[sgpr_32]
v_sub_u32_e32 v[vgpr_B_tid], v0, v[vgpr_tmp]


v_lshrrev_b32_e32 v[vgpr_B_tid_div_4], 2, v[vgpr_B_tid];// /4
v_and_b32_e32     v[vgpr_B_tid_rem_4], 3, v[vgpr_B_tid]; %4

v_lshl_add_u32 v[vgpr_B_read_offset_first],v[vgpr_B_tid_div_4],4,v[vgpr_B_tid_rem_4]

global_load_ushort v[vgpr_B_ushort1], v[vgpr_B_read_offset_first:vgpr_B_read_offset_first+1], s[sgpr_B_addr:sgpr_B_addr+1]


s_waitcnt vmcnt(0)

v_lshlrev_b32_e32 v[vgpr_B_tid_offset_2], 1, v[vgpr_B_tid]

v_mov_b32_e32 v[vgpr_B_ushort1],2.0

;v_mov_b32_e32 v[vgpr_store_addr],v[vgpr_B_tid_offset_2]
v_mov_b32_e32 v[vgpr_store_addr],0
v_mov_b32_e32 v[vgpr_store_addr+1],0
global_store_short v[vgpr_store_addr:vgpr_store_addr+1], v[vgpr_B_ushort1], s[sgpr_kevin_test_float_addr:sgpr_kevin_test_float_addr+1]


s_waitcnt vmcnt(0)

s_mov_b64 exec,s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1]
;vgpr_B_ushort1

;s_barrier;
*/



read_end:
 s_mov_b64 exec,s[sgpr_before_cmp_thread:sgpr_before_cmp_thread+1]

	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw
		.amdhsa_group_segment_fixed_size 24576
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 60
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw, .Lfunc_end0-gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 8620
; NumSgprs: 37
; NumVgprs: 117
; NumAgprs: 128
; TotalNumVgprs: 128
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 24576 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 37
; NumVGPRsForWavesPerEU: 128
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.ident	"HCC clang version 10.0.0 (/data/jenkins-workspace/compute-rocm-rel-3.1/external/hcc-tot/llvm-project/clang 6a70953f87a209f37ea7884abdbb6bcb2d6db732) (based on HCC 3.1.20023-6d267cfb-6a70953f87a )"
	.section	".note.GNU-stack"
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  generic
        .name:           p_wei_global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16
      - .address_space:  generic
        .name:           p_in_global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16
      - .address_space:  generic
        .name:           p_out_global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16

      - .address_space:  generic
        .name:           kevin_int_test
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .value_type:     u32
      - .address_space:  generic
        .name:           kevin_float_test
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16




      - .address_space:  generic
        .name:           p_N
        .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     u32


      - .address_space:  generic
        .name:           p_C
        .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     u32



      - .address_space:  generic
        .name:           p_C0
        .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     u32


      - .address_space:  generic
        .name:           p_hi
        .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .address_space:  generic
        .name:           p_wi
        .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     u32






      - .address_space:  generic
        .name:           datatype_log2
        .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     u32


      - .address_space:  generic
        .name:           load_everytime
        .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     u32

      - .address_space:  generic
        .name:           thread_num
        .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     u32



    .group_segment_fixed_size: 24576
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw
    .private_segment_fixed_size: 0
    .sgpr_count:     64
    .sgpr_spill_count: 0
    .symbol:         gridwise_group_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.kd
    .vgpr_count:     128
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
