# ====== IMPORTS ========
from manim import *
from typing import Callable
from abc import ABC, abstractmethod
from manim.camera.camera import Camera
import numpy as np
import itertools as it

Input2D = float
Input3D = np.ndarray
Output = float
Function2D = Callable[[Input2D], Output]

# TODO: implement fade transitions for everything

class GradientWalkthrough2D(Scene, ABC):
    def __init__(self, 
                 x_range: tuple[float, float, float] = (-10, 10, 1), 
                 y_range: tuple[float, float, float] = (-7, 7, 1),
                 func_equation: str = "f(x)", 
                 x_default: float = 1,
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
        
    def c2p(self, x: Input2D, y: Input2D):
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
        def f(x: Input2D) -> Output:
            x0 = self.x.get_value()
            return self.deriv(x0) * (x - x0) + self.func(x0)
        return f
    
    @abstractmethod
    def func(self, x: Input2D) -> Output:
        """ The function defintion for a given 2D graph """
    
    @abstractmethod
    def deriv(self, x: Input2D) -> Output:
        """ Derivative for the function """
    
    @abstractmethod
    def deriv_2(self, x: Input2D) -> Output:
        """ 2nd derivative for the function """
    

class ParabolaDemo(GradientWalkthrough2D):
    def __init__(self) -> None:
        super().__init__(x_range = (-2, 2, 0.5), y_range = (-1, 4, 0.5), func_equation = "f(x) = x^2", x_default=1, alpha_default=0.9)
        
    def func(self, x: Input2D) -> Output:
        """ The function defintion for a given 2D graph """
        return x ** 2
    
    def deriv(self, x: Input2D) -> Output:
        """ Derivative for the function """
        return 2 * x
    
    def deriv_2(self, x: Input2D) -> Output:
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
        super().__init__(x_range = (-1.2, 2, 0.5), y_range = (-1, 4, 0.5), func_equation = "f(x) = x^4 - 2x^3 + 1.5x")
        self.x.set_value(1.5)
        
    def func(self, x: Input2D) -> Output:
        """ The function defintion for a given 2D graph """
        return x ** 4 - 2 * x ** 3 + 1.5 * x
    
    def deriv(self, x: Input2D) -> Output:
        """ Derivative for the function """
        return 4 * x ** 3 - 6 * x ** 2 + 1.5
    
    def deriv_2(self, x: Input2D) -> Output:
        """ 2nd derivative for the function """
        return 12 * x ** 2 - 12 * x
        
    def construct(self):
        # self.travel_tracker(self.x, [1, 1.5, -0.8, 0.5])
        # self.travel_tracker(self.alpha, [1, 2, 0, 1])
        self.travel_tracker(self.x, [1.5, -1, 1.5])
        self.iterate_gradient_descent()