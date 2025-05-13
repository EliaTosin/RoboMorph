import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

dt = 1/200

radius = 10
time_period = 600
alpha = np.pi/4

w0 = 1/(time_period * dt)
print(w0)
# eef start pos
x0 = 2
y0 = 3

#script settings
num_points = 200

# TODO fare plot cerchio per vedere se va bene e la circonferenza passa da x0, y0

if __name__ == '__main__':

    xs = []
    ys = []

    for t in range(num_points):
        x = radius * math.cos(w0*t + math.pi + alpha) + radius * math.cos(alpha) + x0
        y = radius * math.sin(w0*t + math.pi + alpha) + radius * math.sin(alpha) + y0
        # x = radius * math.cos(w0 * t + alpha) + x0
        # y = radius * math.sin(w0 * t + alpha) + y0

        xs.append(x)
        ys.append(y)

    # Setup plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    traj_line, = ax.plot(xs, ys, 'r.', label='Traiettoria')
    point_marker, = ax.plot([], [], 'bo', label='Posizione attuale')
    ax.plot(x0, y0, 'ko', label='Centro')
    ax.axis('equal')
    ax.legend('right')

    # Slider
    ax_slider = plt.axes([0.15, 0.2, 0.7, 0.03])
    slider = Slider(ax_slider, 'Step', 0, num_points - 1, valinit=0, valstep=1)

    # Bottone Play/Pause
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    button = Button(ax_button, 'Play')
    playing = [False]  # mutable container to allow state change inside callback


    def update(val):
        i = int(slider.val)
        point_marker.set_data(xs[i], ys[i])
        fig.canvas.draw_idle()


    slider.on_changed(update)


    # Frecce sinistra/destra
    def on_key(event):
        val = slider.val
        if event.key == 'left':
            slider.set_val(max(0, val - 1))
        elif event.key == 'right':
            slider.set_val(min(num_points - 1, val + 1))


    fig.canvas.mpl_connect('key_press_event', on_key)


    # Animazione
    def animate(frame):
        if playing[0]:
            val = int(slider.val)
            next_val = (val + 1) % num_points
            slider.set_val(next_val)
            time.sleep(0.25)


    # Play/Pause toggle
    def toggle_play(event):
        playing[0] = not playing[0]
        button.label.set_text("Pause" if playing[0] else "Play")


    button.on_clicked(toggle_play)

    # Avvia animazione
    ani = animation.FuncAnimation(fig, animate, interval=50)

    # Mostra iniziale
    update(0)
    plt.show()
