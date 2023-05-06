from manim import *
from typing import Callable
import numpy as np
import itertools as it

Point = np.ndarray
def gradient_descent_logged(x0: Point, func: Callable[[Point], Point], gradient: Callable[[Point], Point], 
           alpha: float = 0.1, n: int = 5) -> list[Point]:
    """ Computes the next n points given the function and the derivative """
    out = [np.copy(x0)]
    for _ in range(n):
        delta = gradient(x0) * alpha
        x0 -= delta
        out.append(np.copy(x0))
    # making (x,y) rather than just x
    return [np.append(x, func(x)) for x in out]

def main():
    parabola = lambda x: x ** 2
    parabola_gradient = lambda x: 2 * x

    p = gradient_descent_logged(np.array([5.]), parabola, parabola_gradient)
    print(p)

if __name__ == "__main__":
    main()


"""
Point:
    tangent line
    triangle (learning rate, tangent) --> scale alpha
    dy --> how much you change x by
"""

class MinimumForParabola(Scene):
    
    def draw_triangle(self, x: float, a: float, axes: Axes) -> Polygon:
        tangent_func = lambda x_in: 2 * x * (x_in - x) + x ** 2 
        x0, x1 = x - a / 2, x + a / 2
        y0, y1 = tangent_func(x0), tangent_func(x1)
        x2, y2 = [(x1, y0), (x0, y1)][x < 0]
        
        return Polygon(
            axes.c2p(x0, y0),
            axes.c2p(x1, y1),
            axes.c2p(x2, y2)
        )
        
        
    def construct(self):
        x0 = np.array([1.])
        func = lambda x: x ** 2
        derivative = lambda x:  2 * x
        derivative_2 = lambda x: 2

        alpha = 0.8
        points = gradient_descent_logged(x0, func, derivative, alpha=alpha)
        axes = Axes(
            x_range = [-2, 2, 0.5],
            y_range = [-0.5, 2, 0.5],   
        )
        axes_labels = axes.get_axis_labels()
        parabola = axes.plot(func, color=BLUE)

        plot = VGroup(axes, parabola)
        labels = VGroup(axes)
        self.add(plot, labels)

        point = points.pop(0)

        # trackers
        a_tracked = ValueTracker(alpha)
        x_tracked = ValueTracker(point[0])
        # tangent triangle
        tangent_func = lambda x: derivative(point[0]) * (x - point[0]) + point[1]
        dot = always_redraw(lambda: Dot().move_to(axes.c2p(x_tracked.get_value(), func(x_tracked.get_value()))))
        triangle = always_redraw(lambda: self.draw_triangle(x_tracked.get_value(), a_tracked.get_value(), axes))
        self.add(dot)
        self.add(triangle)
            
        # adjust slope
        self.play(x_tracked.animate.set_value(point[0]))
        self.play(x_tracked.animate.set_value(1.5), run_time=2)
        self.play(x_tracked.animate.set_value(0.5), run_time=2)
        self.play(x_tracked.animate.set_value(1), run_time=1)
        
        # adjust learning rate
        self.play(a_tracked.animate.set_value(alpha))
        self.play(a_tracked.animate.set_value(1.3))
        self.play(a_tracked.animate.set_value(0.3))
        self.play(a_tracked.animate.set_value(alpha))
        
        # clear
        self.remove(triangle)

        for p in points:
            # tangent triangle
            tangent_func = lambda x: derivative(point[0]) * (x - point[0]) + point[1]
            x0, x1 = point[0] - alpha / 2, point[0] + alpha / 2
            y0, y1 = tangent_func(x0), tangent_func(x1)
            corner = [(x1, y0), (x0, y1)][derivative(point[0]) * derivative_2(point[0]) < 0] # x1 if derivatives are opp signs
            triangle = Polygon(axes.c2p(x0, y0), axes.c2p(x1, y1), axes.c2p(*corner))
            self.add(triangle)
            self.play(GrowFromPoint(triangle, axes.c2p(*point)))

            # add labels
            # TODO

            # remove triangle (keep dy line)
            corner2 = [(x0, y0), (x1, y1)][derivative(point[0]) * derivative_2(point[1]) > 0] # x0 if derivatives are opp signs
            dy_line = Line(axes.c2p(*corner), axes.c2p(*corner2))
            self.add(dy_line)

            # dy --> move over
            dy_new = Line(axes.c2p(*point), axes.c2p(p[0], point[1]))
            self.play(Transform(dy_line, dy_new))
            # dx --> line down to graph
            to_func = Line(axes.c2p(p[0], point[1]), axes.c2p(*p))
            self.add(to_func)
            self.play(GrowFromPoint(to_func, axes.c2p(p[0], point[1])))

            # moving point (and clearing triangle)
            self.remove(triangle)
            new = Dot(axes.coords_to_point(*p))
            path = axes.plot(func, x_range=[min(p[0], point[0]), max(p[0], point[0])])
            if p[0] < point[0]:
                path.reverse_points()
            self.play(MoveAlongPath(new, path))
            point = p # updating

            # clear 
            self.remove(dy_line)
            self.remove(to_func)

class MinimumForQuartic(Scene):
    """
    Gradient descent with multiple local minima

    x^{4}-2x^{3}+1.5x



    """
