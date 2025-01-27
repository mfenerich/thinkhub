def validate_dependencies(provider: str, req_deps: dict[str, list[str]]):
    """
    Validate that the required dependencies for the specified provider are installed.

    Args:
        provider (str): The name of the provider to validate dependencies for.

    Raises:
        ImportError: If required dependencies are not installed.
    """
    missing_dependencies = []
    for dependency in req_deps.get(provider, []):
        try:
            __import__(dependency)
        except ImportError:
            missing_dependencies.append(dependency)

    if missing_dependencies:
        raise ImportError(
            f"Missing dependencies for provider '{provider}': {', '.join(missing_dependencies)}. "
            f"Install them using 'poetry install --extras {provider}' or 'pip install thinkhub[{provider}]'."
        )