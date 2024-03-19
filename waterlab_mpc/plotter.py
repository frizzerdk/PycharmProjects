"""
Module for visualizing the calculated data.

"""
import matplotlib.pyplot as plt
import numpy as np


def plot_network(network_df, Ts, t, tank_maxdepth, count):

    network_df.plot(subplots=True)

    # fig, axs = plt.subplots(4, 1)
    # axs[0, 0].plot(network_df['tank_inflow'])
    # axs[0, 0].plot(network_df['tank_overflow'])
    # axs[0, 0].set_title('Tank 1 inflow')
    # axs[0, 0].set_ylabel('Flow [LPS]')
    # axs[0, 0].legend(['Inflow', 'Overflow'])
    #
    # axs[1, 0].plot(t, network_df['pump_inflow'])
    # axs[1, 0].set_title('Pump 1 flow')
    # axs[1, 0].set_ylabel('Flow [LPS]')
    #
    # axs[2, 0].plot(t, network_df['tank_volume'])
    # axs[2, 0].set_title('Tank 1 volume')
    # axs[2, 0].set_ylabel('Volume [m3]')
    #
    # axs[3, 0].plot(t, network_df['tank_depth'])
    # axs[3, 0].plot(t, tank_maxdepth[0] * np.ones(count), 'g--')
    # axs[3, 0].set_title('Tank 1 level')
    # axs[3, 0].set_ylabel('Level [m]')
    # axs[3, 0].set_xlabel('Time [%i s]' % Ts)
    # axs[3, 0].legend(['Level', 'Limit'])

    # axs[0, 1].plot(t, [item[1] for item in tank_inflow])
    # axs[0, 1].plot(t, [item[1] for item in tank_overflow])
    # axs[0, 1].set_title('Tank 2 inflow')
    # axs[0, 1].legend(['Inflow', 'Overflow'])
    #
    # axs[1, 1].plot(t, [item[1] for item in pump_flow])
    # axs[1, 1].set_title('Pump 2 flow')
    #
    # axs[2, 1].plot(t, [item[1] for item in tank_volume])
    # axs[2, 1].set_title('Tank 2 volume')
    #
    # axs[3, 1].plot(t, [item[1] for item in tank_depth])
    # axs[3, 1].plot(t, tank_maxdepth[1] * np.ones(count), 'g--')
    # axs[3, 1].set_title('Tank 2 level')
    # axs[3, 1].set_xlabel('Time [%i s]' % Ts)
    # axs[3, 1].legend(['Level', 'Limit'])

    # fig.tight_layout(pad=0.2)
    # for ax in axs.flat:
    #    ax.label_outer()
    # return fig