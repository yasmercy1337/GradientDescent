# ====== IMPORTS ========
from manim import *
from typing import Callable
from manim import np
import numpy as np
import itertools as it
from demo_superclasses import *

class ParabolaDemo(GradientWalkthrough2D):
    def __init__(self) -> None:
        super().__init__(x_range = (-2, 2, 0.5), y_range = (-1, 4, 0.5), func_equation = "f(x) = x^2", x_default=1, alpha_default=0.9)
        
    def func(self, x: Input) -> Output:
        """ The function defintion for a given 2D graph """
        return x ** 2
    
    def deriv(self, x: Input) -> Output:
        """ Derivative for the function """
        return 2 * x
    
    def deriv_2(self, x: Input) -> Output:
        """ 2nd derivative for the function """
        return 2
        
    def construct(self):
        self.always_update_triangle()
        self.always_update_triangle_label()
        self.travel_tracker(self.x, [1, 1.5, -0.5, 0.5])
        self.travel_tracker(self.alpha, [1, 2, 0, 0.9])
        self.iterate_gradient_descent()
        self.iterate_gradient_descent()
        self.play(self.toggle_triangle(), self.toggle_triangle_label())
        self.always_update_triangle()
        self.always_update_triangle_label()
        self.travel_tracker(self.x, [1, 1.5, -0.5, 0.5])
        
class QuarticDemo(GradientWalkthrough2D):
    
    def __init__(self) -> None:
        super().__init__(x_range = (-1.2, 2, 0.5), y_range = (-1, 4, 0.5), func_equation = "f(x) = x^4 - 2x^3 + 1.5x", x_default=1.5)
        
    def func(self, x: Input) -> Output:
        """ The function defintion for a given 2D graph """
        return x ** 4 - 2 * x ** 3 + 1.5 * x
    
    def deriv(self, x: Input) -> Output:
        """ Derivative for the function """
        return 4 * x ** 3 - 6 * x ** 2 + 1.5
    
    def deriv_2(self, x: Input) -> Output:
        """ 2nd derivative for the function """
        return 12 * x ** 2 - 12 * x
        
    def construct(self):
        # self.travel_tracker(self.x, [1, 1.5, -0.8, 0.5])
        # self.travel_tracker(self.alpha, [1, 2, 0, 1])
        # self.travel_tracker(self.x, [1.5, -1, 1.5])
        for _ in range(5):
            self.iterate_gradient_descent()
        
class GaussianDemo(GradientWalkthrough3D):
    
    def __init__(self) -> None:
        super().__init__(func_equation = f"f(x, y) = -e^-(x^2 + y^2)", x_range=[-2, 2], y_range=[-2, 2], z_range=[-1.75, 0.25],
                         x_default=1, y_default=1.2, alpha_default=0.6)
    
    def func(self, x: Input, y: Input) -> Output:
        """ The function defintion for a given 2D graph """
        return -np.exp(- x ** 2 - y ** 2)
    
    def gradient(self, x: Input, y: Input) -> np.ndarray:
        """ Derivative for the function """
        return -2 * self.func(x, y) * np.array([x, y])

    def gradient_2(self, x: Input, y: Input) -> np.ndarray:
        """ Derivative of the gradient """
        f = self.func(x, y)
        return -2 * f * np.array([1 - 2 * x ** 2, 1 - 2 * y ** 2])
    
    def construct(self):
        self.surface.set_opacity(0.3)
        # self.set_camera_orientation(phi=60 * DEGREES, theta=225 * DEGREES,)
        # y = ValueTracker()
        # self.x_partial = always_redraw(lambda: self.get_x_partial(y.get_value()))
        # self.add(self.x_partial)
        # self.travel_tracker(y, [-2, 2, -1])

        # self.add(self.create_tangent_plane())
        # self.add(self.get_x_partial())
        # self.add(self.get_y_partial())
        # self.show_gradient_normal(
        #     self.x, 
        #     0,
        #     camera_thetas=(-90 * DEGREES, 90 * DEGREES),
        # )
        for _ in range(5):
            self.iterate_gradient_descent()


        # plane = always_redraw(self.create_tangent_plane)
        # self.add(plane)
        # self.travel_tracker(self.x, [0.5, -1, 1, 0.3])
        # self.travel_tracker(self.y, [1, -2])

        # phi, theta, *_ =  self.camera.get_value_trackers()
        # self.set_camera_orientation(phi=75 * DEGREES)
        # self.get_y_partial()
        # self.play(theta.animate(run_time=2, rate_func=linear).set_value(theta.get_value() + 360 * DEGREES))

        self.wait()

class LinearRegressionDemo(GradientWalkthrough3D):
    def __init__(self):
        self.n = 100
        self.X, self.Y = self.create_dataset(n=self.n)
        super().__init__(x_range=[-4,4], y_range=[-10, 10], z_range=[-3,3])

    def create_dataset(self, n: int = 50, domain: float = 100, 
                       slope: float = 1, 
                       y_int: float = 5,
                       dev: float = 3) -> tuple[np.ndarray, np.ndarray]:
        """ Returns the X and Y """
        x = (np.random.rand(n) - 0.5) * domain
        r = np.random.normal(0, 1, n) * dev
        y = x * slope - r * dev + y_int
        return x, y

    def linear_func(self, m, b):
        return m * self.X + b

    def func(self, u, v):
        y_pred = self.linear_func(u, v)
        return np.average((y_pred - self.Y) ** 2) / 1000
    
    def gradient(self, m, b):
        y_pred = self.linear_func(m, b)
        diff = y_pred - self.Y

        return 2 / self.n * np.array([
            np.dot(diff, self.X),
            diff
        ])

    def gradient_2(self, m, b):
        return 2 / self.n * np.array([
            np.sum(self.X ** 2),
            np.sum(self.X)
        ])

    def construct(self):
        self.set_camera_orientation(theta=75 * DEGREES, phi=60 * DEGREES)
