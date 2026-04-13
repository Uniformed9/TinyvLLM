from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from dataclasses import dataclass

@dataclass
class ScheduledItem:
    seq: "Sequence"
    is_prefill: bool
    num_tokens: int
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len=config.max_model_len
        self.enable_chunk=config.enable_chunk
        self.eos = config.eos
        
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        
    def schedule(self) ->tuple[list[Sequence],bool]:
        #这一轮给每个seq分多少tokens？
        token_budget=self.max_num_batched_tokens
        req_index=0
        running_seqs=[]
        preempted_seqs = []
    
        scheduled_new_reqs=[]
        while req_index<len(self.running) and token_budget >0:
            seq=self.running[req_index]
            num_new_tokens=len(seq)-seq.num_cached_tokens
            if self.enable_chunk:
                num_new_tokens=min(token_budget,num_new_tokens)
            num_new_tokens=min(num_new_tokens,self.max_model_len-1-seq.num_cached_tokens)
            #分配这么多的new_num_tokens
            # 每一个seq分配
            while not self.block_manager.can_append(seq,num_new_tokens):
                if len(self.running)>req_index:
                    preempted_seq=self.running.pop()
                    self.preempt(preempted_seq)
                    preempted_seqs.append(preempted_seq)
                else:
                    break
            else:
                seq.num_new_tokens=num_new_tokens
                self.block_manager.may_append(seq)
                running_seqs.append(seq)
                token_budget-=seq.num_new_tokens
                req_index+=1
            if len(self.running)<=req_index:
                break
        if not preempted_seqs:
            while self.waiting and token_budget>0 and len(self.running)<self.max_num_seqs:
                seq=self.waiting[0]
                num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens=self.block_manager.get_token_layout(seq)
                if self.enable_chunk:
                    num_new_tokens=min(token_budget,num_new_tokens)
                if num_new_tokens > token_budget or \
                    not self.block_manager.can_allocate(num_new_computed_tokens_in_free + num_new_tokens):
                    break
                seq.num_new_tokens=num_new_tokens
                self.block_manager.allocate(seq)
                assert seq.num_cached_tokens == num_new_computed_tokens_in_free + \
                    num_new_computed_tokens_in_used
                token_budget -= num_new_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_new_reqs.append(seq)
        scheduled_seqs = running_seqs + scheduled_new_reqs
        assert scheduled_seqs
        return scheduled_seqs    
    
    def old_schedule(self) -> tuple[list[Sequence], bool]:
        # 改成chunk prefill 怎么chunk？
        
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], seq_need_compute_logits) -> list[bool]:
        assert len(token_ids) == len(seq_need_compute_logits)
        for seq_index, token_id in zip(seq_need_compute_logits, token_ids):
            seq = seqs[seq_index]
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
                seq.num_completion_tokens == seq.max_tokens or \
                    len(seq) >= self.max_model_len:
                if len(seq) >= self.max_model_len:
                    print(f"Sequence {seq.seq_id} reached max_model_len {self.max_model_len}.")
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens = seq.num_cached_tokens + seq.num_new_tokens
                seq.num_new_tokens = 0
