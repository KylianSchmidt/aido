import b2luigi


class AIDOTask(b2luigi.Task):
    """ Shallow wrapper around b2luigi.Task
    """

    @property
    def htcondor_settings(self):
        return {
            "request_cpus": "1",
            "getenv": "true",
        }
