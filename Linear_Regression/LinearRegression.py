class LinearRegression:

    def __init__(self):
        self.points = []
        
    def fit(self, x, y):
        self.points = list(zip(x, y))

        self.compute_slope()
        print(f'slope: {self.slope}')

        self.compute_y_intercept()
        print(f'y intercept: {self.intercept}')
        

    def predict(self, x_):
        preds = []
        for sample in x_:
            preds.append(self.predict_value(sample))
        return preds

    def SS_maen(self):
        y_ = [y for _, y in self.points]
        mean = sum(y_)/len(y_)

        ss_mean = 0
        for point in y_:
            distance = point - mean
            ss_mean += distance ** 2

        return ss_mean/len(y_)
    
    def SS_fit(self):

        ss_fit = 0
        for point in self.points:
            distance = self.predict_value(point[0]) - point[1] 
            ss_fit += distance ** 2

        return ss_fit / len(self.points)
        
    def R_square(self):
        return (self.SS_maen() - self.SS_fit()) / self.SS_maen()


    def predict_value(self, x):
        return self.slope * x + self.intercept
    
    def mean(self, list):
        return sum(list)/len(list)

    def compute_slope(self):
        divisor = 0
        dividend = 0
        _X = self.mean(list(map(lambda x: x[0], self.points)))
        _Y = self.mean(list(map(lambda x: x[1], self.points)))

        for point in self.points:
            divisor += (point[0] - _X) * (point[1]-_Y)
            dividend += (point[0] - _X) ** 2
            
        self.slope = divisor/dividend

    def compute_y_intercept(self):
        self.intercept = self.mean(list(map(lambda x : x[1], self.points))) - self.mean(list(map(lambda x : x[0], self.points))) * self.slope