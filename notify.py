import os

# brew install terminal-notifier
cmd = "terminal-notifier -message 'done' -activate com.apple.Terminal"

def done():
    os.system(cmd)
