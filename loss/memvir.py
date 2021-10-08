'''
MemVir
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import torch
from queue import Queue

class MemVir():
    def __init__(self, args):
        self.n_classes = args.C
        self.num_step = args.mem_num_step # Hyper-parameter N in MemVir
        self.step_gap = args.mem_step_gap # Hyper-parameter M in MemVir
        self.warm_epoch = args.warm_epoch # Warm-up epoch U_e in MemVir
        self.num_mem = self.num_step * (self.step_gap + 1)

        self.emb_que = [Queue(), Queue()]  # Embedding / labels queue
        self.prx_que = Queue()  # Proxy queue

        self.emb_list = []
        self.target_list = []
        self.proxy_list = []
        self.pxy_len_list = []

    def put_memory(self, output, target, criterion, epoch):
        """ Put proxy, embedding, target into queues

        """

        if epoch <= self.warm_epoch:
            return

        # Put embedding and target into the queue
        target_tensor = target.detach()
        emb_tensor = output.detach()

        self.emb_que[0].put(emb_tensor)
        self.emb_que[1].put(target_tensor)
        while self.emb_que[0].qsize() > self.num_mem:
            self.emb_que[0].get()
            self.emb_que[1].get()

        # Put proxy into the queue
        proxies = criterion.fc_weight.detach().clone()

        self.prx_que.put(proxies)
        while self.prx_que.qsize() > self.num_mem:
            self.prx_que.get()

    def get_memory(self, epoch):
        """ Save memory proxy, embedding, target into lists

        """

        if epoch > self.warm_epoch:
            self.emb_list = list(self.emb_que[0].queue)
            self.target_list = list(self.emb_que[1].queue)
            self.proxy_list = list(self.prx_que.queue)

    def prepare_proxy(self, fc_weight):
        """ Prepare proxy by combining original and memory proxies

        """

        self.pxy_len_list = [fc_weight.shape[1]]
        if len(self.proxy_list) > 0:
            proxy_list = [fc_weight]
            cur_num_step = 0

            if len(self.proxy_list) > self.step_gap:
                self.proxy_list.reverse()
                for idx in range(self.step_gap, len(self.proxy_list), self.step_gap+1):
                    proxy_list.append(self.proxy_list[idx])
                    self.pxy_len_list.append(self.pxy_len_list[cur_num_step] + self.proxy_list[idx].shape[1])
                    cur_num_step += 1

            print("[PM-prx] mem_list: {}, self.num_step: {}, self.step_gap: {}, selected_mem_list: {}".format(
                    len(self.proxy_list), self.num_step, self.step_gap, len(proxy_list)))

            if len(proxy_list) > 1:
                fc_weight = torch.cat(proxy_list, dim=1)

        return fc_weight

    def prepare_emb(self, input, target):
        """ Prepare embedding and target by combining original and memory embeddings

        """

        if len(self.emb_list) > 0:
            input_list = [input]
            target_list = [target]

            cur_num_step = 0
            if len(self.emb_list) > self.step_gap:
                self.emb_list.reverse()
                self.target_list.reverse()
                for idx in range(self.step_gap, len(self.emb_list), self.step_gap+1):
                    input_list.append(self.emb_list[idx])
                    target_list.append(self.target_list[idx] + self.pxy_len_list[cur_num_step])
                    cur_num_step += 1

            print("[PM-emb] mem_list: {}, self.num_step: {}, self.step_gap: {}, selected_mem_list: {}".format(
                    len(self.emb_list), self.num_step, self.step_gap, len(input_list)))

            if len(input_list) > 1:
                input = torch.cat(input_list, dim=0)
                target = torch.cat(target_list, dim=0)

        return input, target
