import json

from bn import BayesianNetwork
import client
from typing import Dict, Union
from fastapi import APIRouter
from pathlib import Path


router = APIRouter()

data_folder = Path("bn/data/")
with open(data_folder / "rectalcancer.json", "r") as file:
    bnFile = file.read().replace("\n", " ")
    bnJson = json.loads(bnFile)


class BayesianNetworkRectalCancer(client.PredictionModelBase):
    result = []

    def initial_start_event(self):
        pass

    def run_calculation(self, model_input: Dict[str, Union[str, int]]):
        Bayesian_network = BayesianNetwork.from_dict(bnJson)
        calculation_results = Bayesian_network.compute_marginals(None, model_input)
        self.result = {
            key: value.zipped() for key, value in calculation_results.items()
        }

        self.post_result(self.result)
        pass

    def static_template_result(self):
        pass


@router.post("/network/_query")
def get_results_from_calculation():
    return {
        "query": [],
        "probabilities": client.PredictionModelStore().get_model_instance().result,
    }

@router.get("/network/rectalcancer")
def get_initial_bn_view():
    return json.loads("""{"json": %s, "id": "rectalcancer"}""" % bnFile)


if __name__ == "__main__":
    client.run(BayesianNetworkRectalCancer, router)
