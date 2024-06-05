
from typing import List, Union, Optional, Set, Tuple
from vllm.executor.gpu_executor import GPUExecutor
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SamplerOutput
from .base import ExecutorGeventBase, make_async


class GPUExecutorAsync(GPUExecutor, ExecutorGeventBase):

    def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output
