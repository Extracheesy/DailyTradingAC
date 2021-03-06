import sys
import os, fnmatch
from load_yfinance_data import _download
from load_yfinance_data import _test1
from load_yfinance_data import _test2
from load_yfinance_data import _get_data_from_csv

from run_strategy import run_trading_strategy

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':

    DOWNLOAD = True
    PROCESS_DATA = True

    if len(sys.argv) > 3:
        if sys.argv[3] == "--noprocessdata":
            print("no process data")
            PROCESS_DATA = False

    if len(sys.argv) > 2:
        if sys.argv[2] == "--nodownload":
            print("no download")
            DOWNLOAD = False

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test1" or sys.argv[1] == "-t1":
            _test1()
        elif sys.argv[1] == "--test2" or sys.argv[1] == "-t2":
            _test2()
        elif sys.argv[1] == "--cac40":
            if(DOWNLOAD == True):
                _download('cac40')
        elif sys.argv[1] == "--apple":
            if (DOWNLOAD == True):
                _download('apple')
            _get_data_from_csv("AAPL")
        else:
            _usage()
    else:
        _usage()




    run_trading_strategy(PROCESS_DATA)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
