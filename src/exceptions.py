class UpliftLibraryError(Exception):
    "Base class"
    pass

class InsufficientDataError(UpliftLibraryError, ValueError):
    "When we do not have enought data"
    pass

class NotFittedError(UpliftLibraryError, RuntimeError):
    "When the model is not fitted"
    pass

class DataValidationError(UpliftLibraryError, ValueError):
    "If we have problems with the data"
    pass
