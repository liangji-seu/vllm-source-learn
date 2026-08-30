class GPUModelRunner:
# init
	max_num_tokens # 本轮最大要计算的tokens数
	max_num_reqs   # 本轮的最大req数
	
	self.kv_caches : list[torch.Tensor] # list[layer_i层的kvcache tensor]
	
	
    #————————————————————————————————————————————————————————————————
	self.requests : dict[req_id, CacheRequestState] # 持久化状态对象库
	self.input_batch # 持久化状态对象池
	#----------------------------------------------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	

	
def execute_model:

	
def _prepare_inputs:

	





# token ids 存储类
    self.input_ids      #   [token_id0, token_id1, .....],  长度=max_num_tokens，表示一个batch的要计算kv的token id的一维缓冲区
    self.inputs_embeds  #   [token_0向量, token_1向量, ...], 长度=max_num_tokens, 表示一个batch的要计算kv的token 向量的一维缓冲区

# token ids 掩码类
    self.is_token_ids   #   [1,1, 0, 0,....], 长度=max_num_tokens， 每一位标志对应位置上的值是否是token id，还是token向量

# token ids kv地址类
    slot_mappings       #   [slot_id, slot_id, .....], 长度=max_num_tokens, 每一位对应这个token的kv要写入的物理地址(KV cache 的线性 slot 编号)。（写kvcache用）


# token 数统计类
	self.num_scheduled_tokens       # [req_0_cal_num, req_0_cal_num, ...], 长度=max_num_reqs， 每一位表示对应req的本轮要计算kv的token数
    self.num_computed_tokens        # [req_0_kv_num, req_1_kv_num, ....], 长度=max_num_reqs，每一位表示对应req的已经计算了kv的token数
    self.num_decode_draft_tokens    # [req_0_draft_num, req_1_draft_num,....], 长度=max_num_reqs, 每一位表示对应req的本轮要验证的草稿token数
    self.prev_num_draft_tokens #[num_draft_0, num_draft_1, ...], 长度=max_num_reqs，每一位表示对应req的上一轮要验证的草稿token数
    self.num_accepted_tokens        # [req_0_accept_num, req_1_accept_num,...], 长度=max_num_reqs， 每一位表示对应req的本轮最终有效输出 token，不只是接受的 draft
    self.seq_lens                   # [req_0_all_num, req_1_all_num, ....], 长度=max_num_reqs，每一位表示对应req的本轮算完后，完成序列长度



# token ids 一维缓冲区位置指示类
                        '''
                        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
                                # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
                                # where M is the max_model_len.
                        '''
    token_indices       # 每个req的token ids列表按照max_model_len填充后，拍成一维，每个input_ids里面的每个token在这个一维填充数组里面的索引
    self.req_indices    # [req_x, req_x, req_y, ...], 长度=max_num_tokens， 每个token属于哪一个req
    self.query_pos      # [req_x_q0, req_x_q1, req_y_q0, ...], 一维token ids中，每个token在各自req内query部分的局部位置
    self.positions      # [pos_0, pos_1, ......], 长度=max_num_tokens， 一维token ids中，每个token在各自req内的绝对位置

    self.query_start_loc #[l_0, l_1, ......, l_n], 长度=max_num_reqs+1, input_ids的区间划分坐标，等同于req_indices



# req_ids 的数组下标映射类
self.prev_positions # [req0_last_pos, req1_last_pos, ...], 长度=max_num_reqs, 本轮的req在上一批的req的InputBatch的_req_ids中的下标