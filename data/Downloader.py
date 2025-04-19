import gdown


class Downloader():

    def __init__(self):
        self.file_urls = {
            "test.en":"1M0ZfBxfCi91arxoGR9yYXxLq0u5o-2pU",
            "test.de":"1FgC9XgvRQTc75yi4kQDhntwJkoRS1Xsm",
            "train.en":"1DsK59SdV-6-rib7mKpuUJgpfRaNxL3YL",
            "train.de":"1awodXSWkS61mkB8jT6hJqEnEUiu72eRT",
            "validation.en":"1G_6aHebLJXNJMTdh8C-oIqG6oypgES1l",
            "validation.de":"1XrOCYwYBCL9NTOHiXxcsW790FRun6JFp",
        }

    def download_file(self, name, file_id):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "data/"+name, quiet=False)

    def download_all_files(self):
        for name, file_id in self.file_urls.items():
            self.download_file(name, file_id)
