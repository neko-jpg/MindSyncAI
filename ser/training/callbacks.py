class EarlyStopping:
    """
    監視指標が一定期間改善しない場合に学習を停止するシンプルなコールバック。
    """

    def __init__(self, patience: int, min_delta: float, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.num_bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience
