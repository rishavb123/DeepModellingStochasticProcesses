import abc


@abc.ABC
class Loader:

    def __init__(self, filename) -> None:
        self.filename = filename

    def load_data() -> None:
        pass

    @abc.abstractmethod
    def download_data(self) -> None:
        pass
