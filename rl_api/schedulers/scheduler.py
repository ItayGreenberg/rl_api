import numpy as np


class LinearScheduler:
    """
    Linearly interpolates a scalar value between `start_value` and `end_value`
    over the inclusive range [start_step, end_step].

    Example
    -------
    sched = LinearScheduler(start_step=0, end_step=1000,
                            start_value=1.0, end_value=0.0)

    # stateless: query any step you like
    lr_at_500 = sched.step(500)

    # stateful: omit the argument to advance internally
    for _ in range(1001):
        lr = sched.step()         # iterates 0→1000 then clamps
    """
    __slots__ = ("start_step", "end_step", "start_value", "end_value",
                 "_span", "_delta", "_cursor")

    def __init__(self, start_step: int, end_step: int,
                 start_value: float, end_value: float):
        if end_step <= start_step:
            raise ValueError("end_step must be > start_step")

        self.start_step  = int(start_step)
        self.end_step    = int(end_step)
        self.start_value = float(start_value)
        self.end_value   = float(end_value)

        self._span   = self.end_step - self.start_step
        self._delta  = self.end_value - self.start_value
        self._cursor = self.start_step  # for state-ful advance

    # -------------------------------------------------------------
    def step(self, step: int | None = None) -> float:
        """
        Parameters
        ----------
        step : int | None
            • int  – return value for that absolute step (stateless mode).  
            • None – use internal cursor and advance it by 1 (stateful mode).

        Returns
        -------
        float
            Interpolated value, clamped to `[start_value, end_value]` outside the
            interpolation range.
        """
        if step is None:
            step = self._cursor
            self._cursor += 1

        # Before range → hold at start_value
        if step <= self.start_step:
            return self.start_value
        # After range → hold at end_value
        if step >= self.end_step:
            return self.end_value

        # Linear interpolation inside range
        frac = (step - self.start_step) / self._span
        return self.start_value + frac * self._delta


