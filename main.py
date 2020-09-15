from modules.train import gram_minimization
from modules.util import get_config

from sys import exit as e

def main():
  configs = get_config("./config.yaml")
  gram_minimization(configs)

if __name__ == '__main__':
  main()