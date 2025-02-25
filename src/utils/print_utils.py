# Text colors
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

# Background colors
BG_RED = '\033[41m'
BG_GREEN = '\033[42m'
BG_YELLOW = '\033[43m'
BG_BLUE = '\033[44m'
BG_PURPLE = '\033[45m'
BG_CYAN = '\033[46m'
BG_WHITE = '\033[47m'

# Styles
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = '\033[0m'  # Resets all styles and colors

def print_red(text):
    """Prints text in red."""
    print(f"{RED}{text}{RESET}")

def print_green(text):
    """Prints text in green."""
    print(f"{GREEN}{text}{RESET}")

def print_yellow(text):
    """Prints text in yellow."""
    print(f"{YELLOW}{text}{RESET}")

def print_blue(text):
    """Prints text in blue."""
    print(f"{BLUE}{text}{RESET}")

def print_purple(text):
    """Prints text in purple."""
    print(f"{PURPLE}{text}{RESET}")

def print_cyan(text):
    """Prints text in cyan."""
    print(f"{CYAN}{text}{RESET}")

def print_white(text):
    """Prints text in white."""
    print(f"{WHITE}{text}{RESET}")

def print_bold(text):
    """Prints text in bold."""
    print(f"{BOLD}{text}{RESET}")

def print_underlined(text):
    """Prints text underlined."""
    print(f"{UNDERLINE}{text}{RESET}")

def print_with_bg_color(text, color):
    """Prints text with a specified background color."""
    print(f"{color}{text}{RESET}")
