class EarlyStopping():
    def __init__(self, patience, crit='min'):
        assert crit in ['min', 'max', 'gan']
        self.patience = patience
        self.crit = crit
        if crit == 'min':
            self.reference = 1e6
        else:
            self.reference = -1
        self.tolerance = 0


    def check(self, step, monitor):
        if self.crit == 'min':
            if monitor < self.reference:
                self.reference = monitor
                self.tolerance = 0
                self.optimum_step = step
            else:
                self.tolerance += 1
        elif self.crit == 'max':
            if monitor > self.reference:
                self.reference = monitor
                self.tolerance = 0
                self.optimum_step = step
            else:
                self.tolerance += 1
        elif self.crit == 'gan':
            if monitor > 0:
                self.tolerance += 1
            else:
                self.tolerance = 0
                self.optimum_step = step

        print (step, monitor, self.reference, self.tolerance)
        if self.tolerance > self.patience:
            if self.crit == 'gan':
                if step > 1000:
                    return self.get_optimum_step()
                else:
                    return None
            else:
                return self.get_optimum_step()
        else:
            return None

    def get_optimum_step(self):
        return self.optimum_step
