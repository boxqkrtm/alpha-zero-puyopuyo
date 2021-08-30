import pyximport; pyximport.install()

import logging

import coloredlogs

from Coach import Coach

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

def main():
    log.info('Loading the Coach...')
    c = Coach()
    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    import sys
    import os
    os.system("chcp 65001")
    main()
