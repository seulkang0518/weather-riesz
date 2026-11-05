import os

# Test reference modules can be imported before the runtime objective wrapper.
# Keep reference math eager regardless of test collection/import order.
os.environ['DRIFT_COMPILE']='0'
