import invertedai as iai
from invertedai.common import AgentAttributes, AgentState 

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random

def main(args):
    init_response = iai.initialize(
        location = args.location,
        states_history = [[AgentState.fromlist([37.08,-31.78,1.52,0.15])]],
        agent_attributes = [AgentAttributes.fromlist([5,2,1.8,'car'])],
    )

    print(f"init_response: {init_response}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--location',
        type=str,
        help=f"IAI formatted map on which to create simulate.",
        default='None'
    )
    args = argparser.parse_args()

    main(args)