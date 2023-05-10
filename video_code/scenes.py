# ====== IMPORTS ========
from manim import *
from typing import Callable
from abc import ABC, abstractmethod
from manim.camera.camera import Camera
import numpy as np
import itertools as it

Input = float
Output = float
Function2D = Callable[[Input], Output]
Function3D = Callable[[Input, Input], Output]

# TODO: implement fade transitions for everything

class GradientWalkthrough3D(ThreeDScene, ABC):
    def __init__(self, 
                 x_range: tuple[float, float, float] = (-6, 6, 1), 
                 y_range: tuple[float, float, float] = (-6, 6, 1),
                 z_range: tuple[float, float, float] = (-6, 6, 1),
                 func_equation: str = "f(x, y)", 
                 x_default: Input = 1,
                 y_default: Input = 1,
                 alpha_default: float = 1,
                 **kwargs
                 ) -> None:
        
        super().__init__(**kwargs)
        
        # setting axis and graph 
        self.ax = ThreeDAxes(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            axis_config={"include_numbers": True},
            x_length=8,
            y_length=6,
            z_length=6
        )
        self.ax_labels = self.ax.get_axis_labels()
        
        self.surface = Surface(
            lambda u, v: self.c2p(u, v, self.func(u, v)),
            u_range=x_range,
            v_range=y_range,
            checkerboard_colors=[BLUE_E],
            resolution=30,
        )
        self.surface_label = MathTex(func_equation, color=BLUE).to_edge(LEFT).to_edge(DOWN)
        self.add(self.ax, self.surface, self.ax_labels)
        
        # values
        self.x = ValueTracker(x_default)
        self.y = ValueTracker(y_default)
        self.alpha = ValueTracker(alpha_default)
        
        # value labels
        self.value_labels = always_redraw(
            lambda: VGroup(
                Tex(f"x = {self.x.get_value():.2f}").to_edge(RIGHT).to_edge(DOWN, buff=1.7),
                Tex(f"y = {self.y.get_value():.2f}").to_edge(RIGHT).to_edge(DOWN, buff=1),
                MathTex(rf"\alpha = {self.alpha.get_value():.2f}").to_edge(RIGHT).to_edge(DOWN)
            )
        )
        
        # visuals
        self.point = always_redraw(self.create_point)
        self.add(self.point)
        
        # camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES,)
        # self.begin_ambient_camera_rotation(rate=PI/6)

        # self.add_fixed_orientation_mobjects(self.surface_label, self.value_labels)
        
    def create_point(self) -> Dot:
        x, y = self.x.get_value(), self.y.get_value()
        return Dot(self.c2p(x, y, self.func(x, y)))
    
    def c2p(self, x: Input, y: Input, z: Output):
        return self.ax.c2p(x, y, z,)
    
    def travel_tracker(self, value: ValueTracker, keypoints: list, runtime: int = 1) -> None:
        for point in keypoints:
            self.play(value.animate(run_time=runtime).set_value(point))
    
    def create_tangent_plane(self, d) -> None:
        x, y = self.x.get_value(), self.y.get_value()
        z = self.func(x, y)
        def tangent_func(u, v):
            dx, dy = self.gradient(x, y)
            return u, v, dx * (u - x) + dy * (v - y) + self.func(x, y)
        
        vertices = [
            self.c2p(*tangent_func(x - d / 2, y - d / 2)),
            self.c2p(*tangent_func(x - d / 2, y + d / 2)),
            self.c2p(*tangent_func(x + d / 2, y + d / 2)),
            self.c2p(*tangent_func(x + d / 2, y - d / 2))
        ]
        return Polygon(*vertices, fill_opacity=0.7, fill_color=BLUE_E)
    
    def get_x_partial(self):
        phi, theta = self.camera.get_phi(), self.camera.get_theta()
        y = self.y.get_value()
        # XZ plane at y
        x_min, x_max, *_ = self.ax.x_range
        z_min, z_max, *_ = self.ax.z_range
        vertices = [
            self.c2p(x_min, y, z_min),
            self.c2p(x_max, y, z_min),
            self.c2p(x_max, y, z_max),
            self.c2p(x_min, y, z_max),
        ]
        yz_plane = Polygon(*vertices, fill_opacity=0.7, fill_color=BLUE_E, color=BLUE_E)
        self.add(yz_plane)
        self.play(Create(yz_plane))
    
        # create z(x) 2D graph
        twoD = self.ax.plot_parametric_curve(
            function=lambda x: np.array([x, y, self.func(x, y)]),
            t_range=self.ax.x_range,
        )
        self.add(twoD)
        self.play(Create(twoD))

        # rotate camera
        self.move_camera(phi=90 * DEGREES, theta=90 * DEGREES * (-1) ** (y < 0))

        # create derivative
        derivative = self.ax.get_secant_slope_group(self.x.get_value(), twoD, dx=0.001)
        self.play(FadeIn(derivative))
        self.wait() 
        
        # clearing / reverting
        self.play(FadeOut(derivative), Uncreate(twoD), Uncreate(yz_plane))
        self.remove(twoD, yz_plane, derivative)
        self.move_camera(phi=phi, theta=theta)

    def get_y_partial(self):
        phi, theta = self.camera.get_phi(), self.camera.get_theta()
        x = self.x.get_value()
        # YZ plane at x
        y_min, y_max, *_ = self.ax.y_range
        z_min, z_max, *_ = self.ax.z_range
        vertices = [
            self.c2p(x, y_min, z_min),
            self.c2p(x, y_max, z_min),
            self.c2p(x, y_max, z_max),
            self.c2p(x, y_min, z_max),
        ]
        yz_plane = Polygon(*vertices, fill_opacity=0.7, fill_color=BLUE_E, color=BLUE_E)
        self.add(yz_plane)
        self.play(Create(yz_plane))
        
        # create z(x) 2D graph
        twoD = self.ax.plot_parametric_curve(
            function=lambda y: [x, y, self.func(x, y)],
            t_range=self.ax.y_range,
        )
        self.add(twoD)
        self.play(Create(twoD))
    
        # rotate camera
        self.move_camera(phi=90 * DEGREES, theta=180 * DEGREES * (x < 0))

        # create derivative
        derivative = self.ax.get_secant_slope_group(self.y.get_value(), twoD, dx=0.001)
        self.play(FadeIn(derivative))
        self.wait()
        # clearing / reverting
        self.play(FadeOut(derivative), Uncreate(yz_plane), Uncreate(twoD))
        self.remove(yz_plane, twoD, derivative)
        self.move_camera(phi=phi, theta=theta)
    
    def create_triangle_dx(self) -> VGroup:
        """ Returns the three lines that form tangent triangle (hor, vert, hypot) """
        x, y, a = self.x.get_value(), self.y.get_value(), self.alpha.get_value()
        tangent = self.tangent_function_dx(x, y)
        opp_signs = self.gradient(x, y)[0] * self.gradient_2(x, y)[0] < 0
        
        x0, x1 = x - a / 2, x + a / 2
        z0, z1 = tangent(x0)[2], tangent(x1)[2]
        x2, z2 = ((x1, z0), (x0, z1))[opp_signs]

        lines = (
            Line(self.c2p(x0, y, z0), self.c2p(x2, y, z2)),
            Line(self.c2p(x1, y, z1), self.c2p(x2, y, z2)),
            Line(self.c2p(x0, y, z0), self.c2p(x1, y, z1)),
        )
        if opp_signs:
            return VGroup(lines[1], lines[0], lines[2])
        return VGroup(*lines)
    
    def create_triangle_dy(self) -> VGroup:
        """ Returns the three lines that form tangent triangle (hor, vert, hypot) """
        x, y, a = self.x.get_value(), self.y.get_value(), self.alpha.get_value()
        tangent = self.tangent_function_dy(x, y)
        opp_signs = self.gradient(x, y)[0] * self.gradient_2(x, y)[0] < 0
        
        y0, y1 = y - a / 2, y + a / 2
        z0, z1 = tangent(y0)[2], tangent(y1)[2]
        y2, z2 = ((y1, z0), (y0, z1))[opp_signs]

        lines = (
            Line(self.c2p(x, y0, z0), self.c2p(x, y2, z2)),
            Line(self.c2p(x, y1, z1), self.c2p(x, y2, z2)),
            Line(self.c2p(x, y0, z0), self.c2p(x, y1, z1)),
        )
        if opp_signs:
            return VGroup(lines[1], lines[0], lines[2])
        return VGroup(*lines)
    
    def tangent_function_dx(self, x0, y0) -> Function3D:
        def f(x: Input) -> Output:
            return x, y0, self.gradient(x0, y0)[0] * (x - x0) + self.func(x0, y0)
        return f

    def tangent_function_dy(self, x0, y0) -> Function3D:
        def f(y: Input) -> Output:
            return x0, y, self.gradient(x0, y0)[1] * (y - y0) + self.func(x0, y0)
        return f

    def show_gradient_normal(self, var: ValueTracker, value: float, 
                             camera_thetas: tuple[float] = (), 
                             dz: float = 0.5):
        """ 
        Demo showing that gradient is the normal vector
        This demo requires there is a point where one of the partials (var) is 0 
        """
        
        def vectors(x: float, y: float) -> VGroup:
            """ NOTE: THIS IS BROKEN WHEN GRADIENT IS 0"""
            dz_dx, dz_dy = self.gradient(x, y)
            z = self.func(x, y)

            # normalize everything to unit
            directions = [
                Vector([1, 0, dz_dx]).get_unit_vector(),      # dx
                Vector([0, 1, dz_dy]).get_unit_vector(),      # dy
            ]
            colors = [RED, GREEN]
            current = np.array([x, y, z])
            vectors = [Arrow(start=self.c2p(*current), end=self.c2p(*(current - v)), color=c) for v, c in zip(directions, colors)]
            return VGroup(*vectors)
        
        def create_normal_vector(x: float, y: float) -> Arrow:
            dz_dx, dz_dy = self.gradient(x, y)
            z = self.func(x, y)
            d = Vector([dz_dx, dz_dy, -1]).get_unit_vector() # normal
            return Arrow(
                start=self.c2p(x, y, z),
                end=self.c2p(x - dz_dx, y - dz_dy, z + 1),
                color=BLUE
            )

        # show graphics then remove vectors
        graphics = vectors(self.x.get_value(), self.y.get_value())
        plane = always_redraw(lambda: self.create_tangent_plane(1))
        normal = always_redraw(lambda: create_normal_vector(self.x.get_value(), self.y.get_value()))
        self.add(graphics, plane, normal)
        self.wait()

        # create optimal direction line
        dx_vector, dy_vector = graphics[0], graphics[1]
            # connect them
        x, y = self.x.get_value(), self.y.get_value()
        dz_dx, dz_dy = self.gradient(x, y)

        def create_arrows(x, y, dz_dx, dz_dy, length):
            z = self.func(x, y)
            def to_length(arr, l):
                return arr * (l / np.sqrt(np.sum(arr ** 2)))
            start = np.array([x, y, z])
            direction_x = to_length(np.array([1, 0, dz_dx]), length)
            direction_y = to_length(np.array([0, 1, dz_dy]), length)

            p0 = start
            p1 = start + direction_x
            p2 = p1 + direction_y

            return VGroup(
                Arrow(start=self.c2p(*p0), end=self.c2p(*p1), color=RED),
                Arrow(start=self.c2p(*p1), end=self.c2p(*p2), color=GREEN),
                Line(start=self.c2p(*p0), end=self.c2p(*p2), color=BLUE)
            )
        
        length = ValueTracker(1)
        arrows = always_redraw(
            lambda: create_arrows(
                self.x.get_value(), 
                self.y.get_value(), 
                *self.gradient(self.x.get_value(), self.y.get_value()), 
                length.get_value()
            )
        )
        self.play(ReplacementTransform(dx_vector, arrows[0]), ReplacementTransform(dy_vector, arrows[1]))
        self.add(arrows)
            # scale up / down (while creating line for passed points)
        self.travel_tracker(length, [0, 1, 0.5])

        # show what happens when move along one var (how normal changes as that var's partial changes)
        self.travel_tracker(self.x, [-1, 0, 1])

    def iterate_gradient_descent(self):
        x0, y0, alpha = self.x.get_value(), self.y.get_value(), self.alpha.get_value()
        dz_dx, dz_dy = self.gradient(x0, y0)
        x1, y1 = x0 - dz_dx * alpha, y0 - dz_dy * alpha
        z0, z1 = self.func(x0, y0), self.func(x1, y1)

        # create triangles
        triangleX = always_redraw(self.create_triangle_dx)
        triangleY = always_redraw(self.create_triangle_dy)
        self.play(Create(triangleX), Create(triangleY))
        # move triangleY to vertex of triangleX
        # TODO
        # move create line vertically to surface
        line = Line(self.c2p(x0, y0, z0), self.c2p(x1, y1, z1))
        self.play(Create(line))
        # move new point
        self.play(self.x.animate.set_value(x1), self.y.animate.set_value(y1))


    @abstractmethod
    def func(self, x: Input, y: Input) -> Output:
        """ The function defintion for a given 2D graph """
    
    @abstractmethod
    def gradient(self, x: Input, y: Input) -> np.ndarray:
        """ Derivative for the function """
    
    @abstractmethod
    def gradient_2(self, x: Input, y: Input) -> np.ndarray:
        """ Derivative of the gradient """
   
class GradientWalkthrough2D(Scene, ABC):
    def __init__(self, 
                 x_range: tuple[float, float, float] = (-10, 10, 1), 
                 y_range: tuple[float, float, float] = (-7, 7, 1),
                 func_equation: str = "f(x)", 
                 x_default: Input = 1,
                 alpha_default: float = 1,
                 **kwargs
                 ) -> None:
        
        super().__init__(**kwargs)
        
        # setting axis and graph 
        self.ax = Axes(
            x_range=x_range,
            y_range=y_range,
            axis_config={"include_numbers": True}
        )
        self.graph = self.ax.plot(self.func, color=BLUE)
        self.graph_label = MathTex(func_equation, color=BLUE).to_edge(LEFT).to_edge(DOWN)
        self.add(self.ax, self.graph, self.graph_label)
        
        # values
        self.x = ValueTracker(x_default)
        self.alpha = ValueTracker(alpha_default)
        
        # value labels
        self.value_labels = always_redraw(
            lambda: VGroup(
                Tex(f"x = {self.x.get_value():.2f}").to_edge(RIGHT).to_edge(DOWN, buff=1),
                MathTex(rf"\alpha = {self.alpha.get_value():.2f}").to_edge(RIGHT).to_edge(DOWN)
            )
        )
        self.add(self.value_labels)
        
        # visuals
        self.point = always_redraw(self.create_point)
        self.add(self.point)
        
        self.triangle = self.create_triangle()
        self.triangle_label = self.create_triangle_label()
        self.show_triangle = False
        self.show_triangle_label = False
        
    def c2p(self, x: Input, y: Output):
        return self.ax.c2p(x, y)
    
    def create_point(self) -> Dot:
        x = self.x.get_value()
        return Dot(self.c2p(x, self.func(x)))
    
    def opp_signs(self) -> bool:
        """ Returns whether concavity matches slope (sign) """
        x = self.x.get_value()
        return self.deriv(x) * self.deriv_2(x) < 0 
    
    def always_update_triangle(self) -> None:
        self.remove(self.triangle)
        self.triangle = always_redraw(self.create_triangle)
        self.add(self.triangle)
        
    def always_update_triangle_label(self) -> None:
        self.remove(self.triangle_label)
        self.triangle_label = always_redraw(self.create_triangle_label)
        self.add(self.triangle_label)
    
    def create_triangle(self) -> VGroup:
        """ Returns the three lines that form tangent triangle (hor, vert, hypot) """
        x, a = self.x.get_value(), self.alpha.get_value()
        tangent = self.tangent_function()
        opp_signs = self.opp_signs()
        
        x0, x1 = x - a / 2, x + a / 2
        y0, y1 = tangent(x0), tangent(x1)
        x2, y2 = ((x1, y0), (x0, y1))[opp_signs]
        
        lines = (
            Line(self.c2p(x0, y0), self.c2p(x2, y2)),
            Line(self.c2p(x1, y1), self.c2p(x2, y2)),
            Line(self.c2p(x0, y0), self.c2p(x1, y1)),
        )
        if opp_signs:
            return VGroup(lines[1], lines[0], lines[2])
        return VGroup(*lines)
    
    def create_triangle_label(self) -> VGroup:
        """ Returns the length measurements and label for horizontal and vertical sides of triangle """
        def create_brace(obj: Mobject, name: str) -> VGroup:
            direction = (DOWN, UP)[self.deriv_2(self.x.get_value()) < 0]
            brace = Brace(obj, direction)
            brace_label = brace.get_tex(name)
            return VGroup(brace, brace_label)
        
        horizontal_side, *_ = self.triangle
        return always_redraw(
            lambda: create_brace(horizontal_side, rf"\alpha = {self.alpha.get_value():.2f}")
        )  
        
    def toggle_triangle_label(self) -> AnimationGroup:
        """ Returns the animations that add/remove the triangle """
        self.show_triangle_label = not self.show_triangle_label and self.show_triangle
        
        if self.show_triangle_label:
            self.triangle_label = self.create_triangle_label()
            self.add(self.triangle_label)
            return AnimationGroup(FadeIn(self.triangle_label))
        else:
            self.remove(self.triangle_label)
            return AnimationGroup(FadeOut(self.triangle_label))
    
    def toggle_triangle(self, keep_vertical: bool = False) -> AnimationGroup:
        """ Returns the animations that fade in/out the triangle """
        self.show_triangle = not self.show_triangle
        
        if self.show_triangle:
            horizontal, vertical, hypotenuse = self.triangle = self.create_triangle()
            self.add(self.triangle)
            return AnimationGroup(
                GrowFromCenter(hypotenuse),
                AnimationGroup(FadeIn(vertical), FadeIn(horizontal))
            )
        elif not keep_vertical:
            self.remove(self.triangle)
            return AnimationGroup(FadeOut(self.triangle))
        self.remove(self.triangle[0], self.triangle[2])
        return AnimationGroup(FadeOut(self.triangle[0]), FadeOut(self.triangle[2]))
    
    def iterate_gradient_descent(self) -> None:
        self.remove(self.triangle)
        self.remove(self.triangle_label)
        
        # new variables
        x0, alpha = self.x.get_value(), self.alpha.get_value()
        x1 = x0 - self.deriv(x0) * alpha
        y0, y1 = self.func(x0), self.func(x1)
        
        # creating triangle
        if not self.show_triangle:
            self.play(self.toggle_triangle(), self.toggle_triangle_label())
        
        # indicating where x1 is
        _, vert, _ = self.triangle
        dx_line = Line(self.c2p(x0, y0), self.c2p(x1, y0))
        dy_line = Line(self.c2p(x1,y0), self.c2p(x1, y1))
        
        # animations
        self.play(self.toggle_triangle(keep_vertical=True), self.toggle_triangle_label())
        self.play(ReplacementTransform(vert, dx_line)) # moving line horizontally
        self.play(GrowFromPoint(dy_line, self.c2p(x1, y0))) # drawing to function
        self.travel_tracker(self.x, [x0, x1]) # moving point
        
        # clearing
        self.remove(dx_line, dy_line)
        
    def travel_tracker(self, value: ValueTracker, keypoints: list) -> None:
        for point in keypoints:
            self.play(value.animate.set_value(point))
    
    def tangent_function(self) -> Function2D:
        def f(x: Input) -> Output:
            x0 = self.x.get_value()
            return self.deriv(x0) * (x - x0) + self.func(x0)
        return f
    
    @abstractmethod
    def func(self, x: Input) -> Output:
        """ The function defintion for a given 2D graph """
    
    @abstractmethod
    def deriv(self, x: Input) -> Output:
        """ Derivative for the function """
    
    @abstractmethod
    def deriv_2(self, x: Input) -> Output:
        """ 2nd derivative for the function """  

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
        self.travel_tracker(self.x, [1.5, -1, 1.5])
        self.iterate_gradient_descent()
        
class GaussianDemo(GradientWalkthrough3D):
    
    def __init__(self) -> None:
        super().__init__(func_equation = f"f(x, y) = e^-(x^2 + y^2)", x_range=[-2, 2], y_range=[-2, 2], z_range=[-0.25, 2],
                         x_default=0.5, y_default=0.5)
    
    def func(self, x: Input, y: Input) -> Output:
        """ The function defintion for a given 2D graph """
        return np.exp(- x ** 2 - y ** 2)
    
    def gradient(self, x: Input, y: Input) -> np.ndarray:
        """ Derivative for the function """
        return -2 * self.func(x, y) * np.array([x, y])

    def gradient_2(self, x: Input, y: Input) -> np.ndarray:
        """ Derivative of the gradient """
        f = self.func(x, y)
        return -2 * f * np.array([1 - 2 * x ** 2, 1 - 2 * y ** 2])
    
    def construct(self):
        self.surface.set_opacity(0.5)
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