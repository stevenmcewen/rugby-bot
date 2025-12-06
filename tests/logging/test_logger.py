import logging

from functions.logging import logger as logger_module


# Does get_logger add a single handler and set the level to INFO?
# must add a single handler and set the level to INFO and return the correct logger
def test_get_logger_adds_single_handler_and_sets_level():
    name = "test.logger.unique"

    log = logger_module.get_logger(name)

    assert isinstance(log, logging.Logger)
    assert log.name == name
    assert log.level == logging.INFO
    handler_count_first = len(log.handlers)
    assert handler_count_first >= 1

    # Calling again for the same name should not add more handlers.
    log_again = logger_module.get_logger(name)
    assert log_again is log
    assert len(log_again.handlers) == handler_count_first


# Set up a dummy logger for testing
class DummyLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        self.info_calls = []
        self.exception_calls = []

    def info(self, msg, *args, **kwargs):
        self.info_calls.append((msg, args, kwargs))

    def exception(self, msg, *args, **kwargs):
        self.exception_calls.append((msg, args, kwargs))


# Does log_function_start use the correct logger and function name and context?
# must use the correct logger and function name and context and call the logger.info method with the correct message and context
def test_log_function_start_uses_info():
    dummy = DummyLogger("dummy")
    logger_module.log_function_start(dummy, "my_func", user="abc")

    assert len(dummy.info_calls) == 1
    msg, args, kwargs = dummy.info_calls[0]
    assert "Starting %s" in msg
    assert args[0] == "my_func"
    assert "context" in kwargs.get("extra", {})
    assert kwargs["extra"]["context"]["user"] == "abc"


# Does log_function_error use the correct logger and function name and exception?
# must use the correct logger and function name and exception and call the logger.exception method with the correct message and exception
def test_log_function_error_uses_exception():
    dummy = DummyLogger("dummy")
    exc = RuntimeError("boom")

    logger_module.log_function_error(dummy, "my_func", exc)

    assert len(dummy.exception_calls) == 1
    msg, args, _ = dummy.exception_calls[0]
    assert "Error in %s" in msg
    assert args[0] == "my_func"
    assert args[1] is exc


