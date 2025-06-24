import logging

# Setup the package-level logger
logging.getLogger("colboost").addHandler(logging.NullHandler())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
