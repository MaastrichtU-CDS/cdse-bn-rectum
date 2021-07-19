import json

from bn import BayesianNetwork
import client
from typing import Dict, Union
from fastapi import APIRouter


router = APIRouter()


class BayesianNetworkRectalCancer(client.PredictionModelBase):
    result = []

    def initial_start_event(self):
        pass

    def run_calculation(self, model_input: Dict[str, Union[str, int]]):
        with open("bn\\data\\rectalcancer.json", "r") as file:
            bnFile = file.read().replace("\n", " ")
            bnJson = json.loads(bnFile)

        Bayesian_network = BayesianNetwork.from_dict(bnJson)
        calculation_results = Bayesian_network.compute_marginals(None, model_input)
        self.result = {
            key: value.zipped() for key, value in calculation_results.items()
        }
        print(self.result)
        self.post_result(self.result)
        pass

    def static_template_result(self):
        pass


@router.get("/_query")
def example_extra_route():
    return client.PredictionModelStore().get_model_instance().result


if __name__ == "__main__":
    client.run(BayesianNetworkRectalCancer, router)
