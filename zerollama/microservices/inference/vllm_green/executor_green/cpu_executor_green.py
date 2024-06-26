from typing import List, Union, Optional, Set, Tuple
from vllm.executor.cpu_executor import CPUExecutor
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SamplerOutput
from .base import ExecutorGeventBase, make_async


class CPUExecutorAsync(CPUExecutor, ExecutorGeventBase):

    def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output

    def check_health_async(self) -> None:
        # CPUExecutor will always be healthy as long as
        # it's running.
        return