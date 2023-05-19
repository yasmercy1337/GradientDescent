# ====== IMPORTS ========
from manim import *
from typing import Callable
from manim import np
import numpy as np
import itertools as it
from demo_superclasses import *

class ParabolaDivergence(GradientWalkthrough2D):
    def __init__(self) -> None:
        super().__init__(x_range = (-2, 2, 0.5), y_range = (-1, 4, 0.5), func_equation = "f(x) = x^2", x_default=0.5, alpha_default=1.1)
        
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
        for _ in range(5):
            self.iterate_gradient_descent()
        
class QuarticLocalMinima(GradientWalkthrough2D):
    
    def __init__(self) -> None:
        super().__init__(x_range = (-1.2, 2, 0.5), y_range = (-1, 4, 0.5), func_equation = "f(x) = x^4 - 2x^3 + 1.5x", 
                         x_default=0.8, alpha_default=0.3)
        
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
        self.n = 50
        self.X, self.Y = create_dataset(n=self.n, domain=100, slope=0.7, y_int=0)
        super().__init__(x_range=[0, 1.8], y_range=[-30, 30, 15], z_range=[0, 8, 2],
                         x_default=1.5, y_default=-25, alpha_default=0.5)
        self.set_camera_orientation(theta=4 * DEGREES, phi=80 * DEGREES)

    def linear_func(self, m, b):
        return m * self.X + b

    def func(self, u, v):
        y_pred = self.linear_func(u, v)
        return np.sum((y_pred - self.Y) ** 2) / 10_000
    
    def gradient(self, m, b):
        y_pred = self.linear_func(m, b)
        diff = y_pred - self.Y
        
        return 2 / 10_000 * np.array([
            np.dot(diff, self.X),
            np.sum(diff)
        ])

    def gradient_2(self, m, b):
        return 2 / 10_000 * np.array([
            m * np.sum(self.X ** 2),
            self.n,
        ])

    def iterate_gradient_descent(self):
        # variables
        x, y, a = self.x.get_value(), self.y.get_value(), self.alpha.get_value()
        z = self.func(x, y)
        gradient = self.gradient(x, y)
        dx, dy = gradient * a / np.sqrt(np.sum(gradient ** 2))
        x1, y1 = x - dx, y - dy
        z1 = self.func(x1, y1)
        
        # get the 2D functions where x and y are fixed respectively
        y_slice_f = lambda t: [t, y, self.func(t, y)]
        x_slice_f = lambda t: [x, t, self.func(x, t)]
        y_tangent_f = self.tangent_function_dx(x, y)
        x_tangent_f = self.tangent_function_dy(x, y)
        
        y_slice = self.ax.plot_parametric_curve(y_slice_f, t_range=self.ax.x_range)
        x_slice = self.ax.plot_parametric_curve(x_slice_f, t_range=self.ax.y_range)
        y_tangent_range = np.array([self.ax.z_range[0] - z, self.ax.z_range[1] - z]) / dx + x
        x_tangent_range = np.array([self.ax.z_range[0] - z, self.ax.z_range[1] - z]) / dy + y
        y_tangent = self.ax.plot_parametric_curve(y_tangent_f, t_range=[min(y_tangent_range), max(y_tangent_range)])
        x_tangent = self.ax.plot_parametric_curve(x_tangent_f, t_range=[min(x_tangent_range), max(x_tangent_range)])
        
        # creating tangent and slices
        self.play(Create(y_slice), Create(x_slice))
        self.play(Create(x_tangent), Create(y_tangent))
        
        # remove slices
        self.play(FadeOut(x_slice), FadeOut(y_slice))
        
        # convert tangent lines to vectors of length alpha (that are pointing down)
        dir_dx = np.array([-dx, 0, -dx ** 2])
        dir_dy = np.array([0, -dy, -dy ** 2])
        p0, p1, p2, p3 = [x, y, z], [x, y, z] + dir_dx, [x, y, z] + dir_dy, [x, y, z] + dir_dx + dir_dy
        print("testing")
        assert np.sum(p3[:2] - [x1, y1]) < 0.05
        
        vdx = Arrow(start=self.c2p(*p0), end=self.c2p(*p1))
        vdy = Arrow(start=self.c2p(*p0), end=self.c2p(*p2))
        self.play(ReplacementTransform(y_tangent, vdx), ReplacementTransform(x_tangent, vdy))
        
        # add vectors
        v3 = Arrow(start=self.c2p(*p2), end=self.c2p(*p3))
        self.play(ReplacementTransform(vdx, v3))

        # add line going to function
        line = Arrow(start=self.c2p(*p3), end=self.c2p(x1, y1, z1))
        self.play(GrowFromPoint(line, self.c2p(x1, y1, z1)))
        
        self.play(self.x.animate.set_value(x1), self.y.animate.set_value(y1)) # move
        self.remove(vdx, vdy, v3, line) # clear
        
    def construct(self):
        self.surface.set_opacity(0.5)

        for _ in range(3):
            self.iterate_gradient_descent()
            print(self.x.get_value(), self.y.get_value())
        self.wait(1)

    def iterate(self):
        x, y, a = self.x.get_value(), self.y.get_value(), self.alpha.get_value()
        n = 2
        for i in range(n):
            # if i % (n // 10) == 0:
            print(f"iteration {i}", x, y, self.func(x, y))
                
            dx, dy = self.gradient(x, y) * a
            x, y = x - dx, y - dy
   
class FirstDerivativeTest(Scene):
    
    def r(self, t: float) -> float:
        return 20 * np.sin(t ** 2 / 35)
    
    def d(self, t: float) -> float:
        return -0.04 * t ** 3 + 0.4 * t ** 2 + 0.96 * t
    
    def dt(self, t: float) -> float:
        return self.r(t) - self.d(t)
    
    def f(self, t: float) -> float:
        if t == 0:
            return 30
        
        dx = t / 1000
        x = np.arange(0, t, dx)
        return 30 + np.trapz(self.dt(x), dx=dx)
    
    def construct(self):
        x_range = [0, 8, 1]
        y_range = [20, 80, 10]
        
        # display graph
        axes = Axes(x_range=x_range, y_range=y_range, axis_config={"include_numbers": True})
        axes_labels = axes.get_axis_labels()
        graph = axes.plot(self.f, x_range=x_range)
        self.add(axes, axes_labels, graph)
        
        self.wait(1) # this is function
        
        # move tangent line and keep track of slope, note when derivative = 0 (with dot)
        t = ValueTracker(0)
        dot = always_redraw(lambda: Dot(axes.c2p(t.get_value(), self.f(t.get_value()))))
        tangent_func = lambda x: (self.dt(t.get_value())) * (x - t.get_value()) + self.f(t.get_value())
        tangent = always_redraw(lambda: axes.plot(tangent_func, x_range=x_range))
        slope_label = always_redraw(lambda: Tex(f"f'({t.get_value():.2f}) = {self.dt(t.get_value()):.2f}")
                                    .to_edge(RIGHT).to_edge(UP, buff=1))
        self.add(dot, tangent, slope_label)
        self.play(FadeIn(tangent), FadeIn(slope_label))
        
        points, coords = [], []
        # moving through critical points      
        critical_points = [0, 3.272, 8]
        for p in critical_points:
            self.play(t.animate.set_value(p)) # move to point
            x, y = t.get_value(), self.f(t.get_value())
            point = Dot(axes.c2p(x, y))
            coord = Tex(f"({x:.2f}, {y:.2f})").move_to(axes.c2p(min(7.5, x + 0.5), y + 5))
            self.play(Create(point), Write(coord)) # add
            
            coords.append(coord)
            points.append(point)
        
        self.play(FadeOut(tangent), FadeOut(slope_label))
        
        # change text for global minimum
        x, y = critical_points[1], self.f(critical_points[1])
        # new_text = Tex(r"\textbf" + f"{{({x:.2f}, {y:.2f})}}").move_to(axes.c2p(min(7.5, x + 0.5), y + 5)) # bold text
        new_text = Tex(f"({x:.2f}, {y:.2f})", color=RED).move_to(axes.c2p(min(7.5, x + 0.5), y + 5)) # red text
        self.play(ReplacementTransform(coords[1], new_text))
        
        self.wait(1)

class LeastSquaresCost(ThreeDScene):
    
    def create_square(self, x: float, y_pred: float, y_real: float, axes: Axes) -> Polygon:
        residual = y_pred - y_real
        return Polygon(
            axes.c2p(x, y_real),
            axes.c2p(x, y_pred),
            axes.c2p(x + residual, y_pred),
            axes.c2p(x + residual, y_real)
        )
    
    def create_all_squares(self, x, y_pred, y_real, axes) -> VGroup:
        return VGroup(*(self.create_square(*data, axes) for data in zip(x, y_pred, y_real)))
    
    def linear(self, m: float, b: float):
        return lambda x: m * x + b
    
    def residuals(self, y_pred, y_real):
        return np.sum((y_pred - y_real) ** 2)
    
    def cost(self, x, m, b, y_real):
        y_pred = m * x + b
        return self.residuals(y_pred, y_real)
    
    def construct(self):
        m = ValueTracker(0.7)
        b = ValueTracker(0)
        x, y_real = create_dataset(domain=100, slope=0.7, y_int=0)
        
        x_range=[-90, 90, 30]
        y_range=[-60, 60, 30]
        axes = Axes(x_range=x_range, y_range=y_range, axis_config={"include_numbers": True}, x_length=12, y_length=8)
        points = VGroup(*(Dot(axes.c2p(*p)) for p in zip(x, y_real)))
        self.add(axes)
        self.play(FadeIn(points))
        self.add(points)
        
        best_fit = always_redraw(lambda: axes.plot(self.linear(m.get_value(), b.get_value()), x_range=x_range))
        squares = always_redraw(lambda: self.create_all_squares(x, m.get_value() * x + b.get_value(), y_real, axes))
        rss_label = always_redraw(lambda: Tex(f"Residual Squared Sum: {self.cost(x, m.get_value(), b.get_value(), y_real):.2f}")
                                  .to_edge(LEFT).to_edge(UP, buff=1))

        # create given line then the squares
        self.play(Create(best_fit))
        self.play(FadeIn(squares), Write(rss_label))
        self.add(best_fit, squares, rss_label)
        
        # shift this axis to the side
        new_axes = Axes(x_range=x_range, y_range=y_range, axis_config={"include_numbers": True}, x_length=6, y_length=4).to_edge(RIGHT)
        new_points = VGroup(*(Dot(new_axes.c2p(*p)) for p in zip(x, y_real)))
        self.play(AnimationGroup(ReplacementTransform(axes, new_axes), ReplacementTransform(points, new_points), run_time=2))
        self.add(new_points, new_axes)
        
        # create second graph for cost given biases 
        x_range=[-60, 60, 30]
        y_range=[0, 20, 2]
        y_scale = 0.0001 # 1 / 10_000
        cost_axes = Axes(x_range=x_range, y_range=y_range, axis_config={"include_numbers": True}, 
                         x_length=6, y_length=4, tips=False).to_edge(LEFT)
        self.add(cost_axes)
        self.play(Create(cost_axes))
        
        # create the cost graph from cost = -60 to cost = 60
        cost_graph = always_redraw(lambda: cost_axes.plot(
            lambda b: self.cost(x, m.get_value(), b, y_real) * y_scale, 
            x_range=[-60, b.get_value(), 30]))
        self.play(b.animate(run_time=3).set_value(-60))
        self.add(cost_graph)
        self.play(b.animate(run_time=5).set_value(60))
        self.play(m.animate(run_time=2).set_value(0.1))
        
        # change to 3D graph
        x_range=[-60, 60, 30] # bias
        y_range=[0, 20, 2] # cost
        z_range=[0, 2, 1] # slope
        z_scale = 0.0001 # 1 / 10_000
        cost_axes_3D = ThreeDAxes(x_range=x_range, y_range=y_range, z_range=z_range, axis_config={"include_numbers": True}, 
                                  x_length=6, y_length=4, tips=False).to_edge(LEFT)
        
        cost_graph_3D = always_redraw(lambda: VGroup(*(
            cost_axes_3D.plot_parametric_curve(
                lambda b: [b, self.cost(x, m, b, y_real) * z_scale, m],
                t_range=x_range)
            for m in np.arange(0, m.get_value(), 0.1))
        ))
        
        self.play(FadeIn(cost_axes_3D), FadeOut(cost_axes))
        self.play(ReplacementTransform(cost_graph, cost_graph_3D))
        self.add(cost_graph_3D, cost_axes_3D)
        self.play(b.animate(run_time=1).set_value(0))
        self.play(m.animate(run_time=3).set_value(2))
        
        self.wait(2)

def create_dataset(
    n: int = 50, 
    domain: float = 100, 
    slope: float = 1, 
    y_int: float = 5,
    dev: float = 3
    ) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the X and Y """
    np.random.seed(1337)
    
    x = (np.random.rand(n) - 0.5) * domain
    r = np.random.normal(0, 1, n) * dev
    y = x * slope - r * dev + y_int
    return x, y

if __name__ == "__main__":
    x, y = create_dataset(n=50, domain=100, slope=0.7, y_int=0)
    x = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(x, y, rcond=None)[0]
    print("Best:", m, b, LinearRegressionDemo().func(m, b))
    # print(LinearRegressionDemo().func(m, b))
    # print(LinearRegressionDemo().func(0.6, 1))
    LinearRegressionDemo().iterate()
    