import random
import time
import turtle


class SnakeGame:
    __BG_COLOR = [211 / 255, 228 / 255, 211 / 255]

    __HEAD_COLOR = [24 / 255, 29 / 255, 26 / 255]

    __FOOD_COLOR = [124 / 255, 132 / 255, 119 / 255]

    def __init__(self):

        self.__delay = 0.1

        self.__stop = False

        # Score
        self.__score = 0
        self.__high_score = 0

        # Set up the screen
        self.__game_window = turtle.Screen()
        self.__game_window.title("py-snake")
        self.__game_window.bgcolor(*SnakeGame.__BG_COLOR)
        self.__game_window.setup(width=600, height=600)
        self.__game_window.tracer(0)  # Turns off the screen updates
        root = turtle.getcanvas().winfo_toplevel()
        root.iconbitmap("assets/snake.ico")

        # Snake head
        self.__head = turtle.Turtle()
        self.__head.speed(0)
        self.__head.shape("square")
        self.__head.color(*SnakeGame.__HEAD_COLOR)
        self.__head.penup()
        self.__head.goto(0, 0)
        self.__head.direction = "stop"

        # Snake food
        self.__food = turtle.Turtle()
        self.__food.speed(0)
        self.__food.shape("square")
        self.__food.color(*SnakeGame.__FOOD_COLOR)
        self.__food.penup()
        self.__food.goto(0, 100)

        self.__segments = []

        # Score
        self.__pen = turtle.Turtle()
        self.__pen.speed(0)
        self.__pen.shape("square")
        self.__pen.color(*SnakeGame.__HEAD_COLOR)
        self.__pen.penup()
        self.__pen.hideturtle()
        self.__pen.goto(0, 260)

        self.__refresh_score()

    # Functions
    def go_up(self):
        if self.__head.direction != "down":
            self.__head.direction = "up"

    def go_down(self):
        if self.__head.direction != "up":
            self.__head.direction = "down"

    def go_left(self):
        if self.__head.direction != "right":
            self.__head.direction = "left"

    def go_right(self):
        if self.__head.direction != "left":
            self.__head.direction = "right"

    def do_nothing(self):
        pass

    def __refresh_score(self):
        self.__pen.write("Score: {}  High Score: {}".format(self.__score, self.__high_score),
                         align="center",
                         font=("System", 24, "normal"))

    def __move(self):
        if self.__head.direction == "up":
            y = self.__head.ycor()
            self.__head.sety(y + 20)

        if self.__head.direction == "down":
            y = self.__head.ycor()
            self.__head.sety(y - 20)

        if self.__head.direction == "left":
            x = self.__head.xcor()
            self.__head.setx(x - 20)

        if self.__head.direction == "right":
            x = self.__head.xcor()
            self.__head.setx(x + 20)

    def set_stop(self):
        self.__stop = True

    def __reset(self):
        time.sleep(1)

        self.__head.goto(0, 0)
        self.__head.direction = "stop"

        # Hide the segments
        for segment in self.__segments:
            segment.goto(1000, 1000)

        # Clear the segments list
        self.__segments.clear()

        # Reset the score
        self.__score = 0

        # Reset the delay
        self.__delay = 0.1

        # Update the score display
        self.__pen.clear()
        self.__refresh_score()

        self.__stop = False

    # Main game loop
    def run(self):

        while True:
            self.__game_window.update()

            # Check for a collision with the border
            if self.__head.xcor() > 290:
                self.__head.goto(-290, self.__head.ycor())

            elif self.__head.xcor() < -290:
                self.__head.goto(290, self.__head.ycor())

            elif self.__head.ycor() > 290:
                self.__head.goto(self.__head.xcor(), -290)

            elif self.__head.ycor() < -290:
                self.__head.goto(self.__head.xcor(), 290)

            # Check for a collision with the food
            if self.__head.distance(self.__food) < 20:

                # Move next food to a random spot
                x = random.randint(-290, 290)
                y = random.randint(-290, 290)
                self.__food.goto(x, y)

                # Add a segment
                new_segment = turtle.Turtle()
                new_segment.speed(0)
                new_segment.shape("square")
                new_segment.color(*SnakeGame.__FOOD_COLOR)
                new_segment.penup()
                self.__segments.append(new_segment)

                # Shorten the delay
                self.__delay -= 0.001

                # Increase the score
                self.__score += 10

                if self.__score > self.__high_score:
                    self.__high_score = self.__score

                self.__pen.clear()

                self.__refresh_score()

            # Move the end segments first in reverse order
            for index in range(len(self.__segments) - 1, 0, -1):
                x = self.__segments[index - 1].xcor()
                y = self.__segments[index - 1].ycor()
                self.__segments[index].goto(x, y)

            # Move segment 0 to where the head is
            if len(self.__segments) > 0:
                x = self.__head.xcor()
                y = self.__head.ycor()
                self.__segments[0].goto(x, y)

            self.__move()

            # Check for head collision with the body segments
            for segment in self.__segments:
                if segment.distance(self.__head) < 20:
                    self.__reset()

            if self.__stop is True:
                self.__reset()

            time.sleep(self.__delay)
