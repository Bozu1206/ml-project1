class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
def print_greet():
    print(style.WHITE + "ML Project 1 --- Heart Disease Prediction" + style.RESET)

def print_loading_data():
    print(style.YELLOW + "=================================================================================" + style.RESET, flush=True)
    print(style.YELLOW + "Loading data ... " + style.RESET, end = " ", flush=True)

def print_finish_loading_data(time):
    print(style.YELLOW + f"is Finished. It took {time:.2f} seconds" + style.RESET)
    print(style.YELLOW + "=================================================================================" + style.RESET)
    
def print_cv_lauching_g(): 
    print(style.RED + "=================================================================================" + style.RESET)
    
def print_cv_lauching(param, name_of_model):
    print(style.WHITE + f"Cross validation for {name_of_model} for parameter {param}..." + style.RESET)
    
def print_cv_best_param(param, score): 
    print(style.WHITE + f"\tBest parameter found: {param} (f1: {score:0.4f})..." + style.RESET)
    
def print_cv_ending():
    print(style.RED + "=================================================================================" + style.RESET, flush=True)
    
def print_start_best_models():
    print(style.BLUE + "=================================================================================" + style.RESET)
    print(style.BLUE + "Training model ..." + style.RESET)
    
def print_end_best_models():    
    print(style.BLUE + "... is Finished" + style.RESET)
    print(style.BLUE + "=================================================================================" + style.RESET)
    