class IRPError(Exception):
    """Base error for this project."""
    pass

class ConfigError(IRPError):
    pass

class DataError(IRPError):
    pass

class APIError(IRPError):
    pass



"""
Gleicher code, verschiedene Reaktionen!

    try:
        load_config()
        load_data()
        call_api()
    except APIError:
        print("API kaputt")
    except DataError:
        print("Daten kaputt")
    except ConfigError:
        print("Config kaputt")


Fehlerhierarchie:

IRPError
 ├── ConfigError
 ├── DataError
 └── APIError
 
    """
