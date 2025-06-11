import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def moving_average(data, window_size=10):
    """
    Calculate moving average with specified window size.
    
    Args:
        data (list or numpy.ndarray): Input data array
        window_size (int): Size of the moving window
        
    Returns:
        numpy.ndarray: Moving average of the data
    """
    data_array = np.array(data)
    cumsum = np.cumsum(np.insert(data_array, 0, 0))
    
    # Calculate moving average with proper handling for beginning of array
    ma = np.zeros_like(data_array, dtype=float)
    
    for i in range(len(data_array)):
        if i < window_size:
            # For the beginning, use available data only
            ma[i] = cumsum[i+1] / (i+1)
        else:
            # For the rest, use the window
            ma[i] = (cumsum[i+1] - cumsum[i+1-window_size]) / window_size
    
    return ma


def plot_reward(ax, reward_history, window_size=10):
    """
    Plot reward history and its moving average.
    
    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on
        reward_history (list or numpy.ndarray): Reward history
        window_size (int): Window size for moving average
    """
    # Convert to numpy array if it's not already
    rewards = np.array(reward_history)
    
    # Calculate timesteps
    timesteps = np.arange(len(rewards))
    
    # Calculate moving average
    avg_rewards = moving_average(rewards, window_size)
    
    # Plot raw rewards
    ax.plot(timesteps, rewards, 'b-', alpha=0.4, label='Rewards')
    
    # Plot moving average
    ax.plot(timesteps, avg_rewards, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    # Add labels and grid
    ax.set_title('Reward History')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()


def plot_acceleration(ax, acceleration_history, title, components=('X', 'Y', 'Z')):
    """
    Plot acceleration history (linear or angular) with statistics.
    
    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on
        acceleration_history (list or numpy.ndarray): Acceleration history
                                                    shape: (timesteps, 3)
        title (str): Title for the plot
        components (tuple): Labels for the components (default: X, Y, Z)
    """
    # Convert to numpy array if it's not already
    accel = np.array(acceleration_history)
    
    # Calculate timesteps
    timesteps = np.arange(len(accel))
    
    # Plot each component
    colors = ['r', 'g', 'b']
    for i in range(accel.shape[1]):
        ax.plot(timesteps, accel[:, i], f'{colors[i]}-', linewidth=0.25, alpha=0.7, 
                label=f'{components[i]}')
    
    # Calculate statistics
    means = np.mean(accel, axis=0)
    stds = np.std(accel, axis=0)
    
    # Create statistics text
    stats_text = '\n'.join([
        f"Statistics:",
        f"{components[0]}: mean={means[0]:.4f}, std={stds[0]:.4f}",
        f"{components[1]}: mean={means[1]:.4f}, std={stds[1]:.4f}",
        f"{components[2]}: mean={means[2]:.4f}, std={stds[2]:.4f}"
    ])
    
    # Add statistics text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Add labels and grid
    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Acceleration')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()


def plot_simulation_history(reward_history, linear_acceleration_history, 
                           angular_acceleration_history, window_size=10, 
                           figsize=(12, 10), save_path=None):
    """
    Plot simulation history including rewards and accelerations.
    
    Args:
        reward_history (list or numpy.ndarray): Reward history
        linear_acceleration_history (list or numpy.ndarray): Linear acceleration history
                                                           shape: (timesteps, 3)
        angular_acceleration_history (list or numpy.ndarray): Angular acceleration history
                                                            shape: (timesteps, 3)
        window_size (int): Window size for moving average of rewards
        figsize (tuple): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure. If None, the figure is shown.
    """
    # Create figure and grid for subplot layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.4)
    
    # Create subplots
    ax_reward = fig.add_subplot(gs[0])
    ax_linear = fig.add_subplot(gs[1])
    ax_angular = fig.add_subplot(gs[2])
    
    # Plot reward history
    plot_reward(ax_reward, reward_history, window_size)
    
    # Plot linear acceleration
    plot_acceleration(ax_linear, linear_acceleration_history, 
                     'Linear Acceleration')
    
    # Plot angular acceleration
    plot_acceleration(ax_angular, angular_acceleration_history, 
                     'Angular Acceleration')
    
    # Adjust layout to make room for text boxes
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_each_separately(reward_history, linear_acceleration_history, 
                         angular_acceleration_history, window_size=10, 
                         figsize=(10, 6), save_dir=None):
    """
    Create separate plots for rewards and accelerations.
    
    Args:
        reward_history (list or numpy.ndarray): Reward history
        linear_acceleration_history (list or numpy.ndarray): Linear acceleration history
                                                           shape: (timesteps, 3)
        angular_acceleration_history (list or numpy.ndarray): Angular acceleration history
                                                            shape: (timesteps, 3)
        window_size (int): Window size for moving average of rewards
        figsize (tuple): Figure size (width, height) in inches
        save_dir (str, optional): Directory to save figures. If None, figures are shown.
    """
    # Plot rewards
    fig_reward = plt.figure(figsize=figsize)
    ax_reward = fig_reward.add_subplot(111)
    plot_reward(ax_reward, reward_history, window_size)
    
    if save_dir:
        plt.savefig(f"{save_dir}/rewards.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot linear acceleration
    fig_linear = plt.figure(figsize=figsize)
    ax_linear = fig_linear.add_subplot(111)
    plot_acceleration(ax_linear, linear_acceleration_history, 'Linear Acceleration')
    
    if save_dir:
        plt.savefig(f"{save_dir}/linear_acceleration.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot angular acceleration
    fig_angular = plt.figure(figsize=figsize)
    ax_angular = fig_angular.add_subplot(111)
    plot_acceleration(ax_angular, angular_acceleration_history, 'Angular Acceleration')
    
    if save_dir:
        plt.savefig(f"{save_dir}/angular_acceleration.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig_reward, fig_linear, fig_angular


def plot_smooth_acceleration_data():
    # Parsed data from the text
    linear_acc = [
        [-2.78958594e-03,  1.04286519e-08, -1.99502211e-01],
        [ 5.70924385e-02, -4.22441636e-04,  6.45962536e-01],
        [-0.02887852, -0.00708242, -0.03441953],
        [-0.00241784,  0.00786155,  0.0115559 ],
        [ 0.02852874, -0.01809564, -0.01578597],
        [ 0.03418262,  0.01512451,  0.029557  ],
        [ 0.01669935,  0.00192257, -0.03044287],
        [ 0.05098682, -0.0349929 ,  0.0325957 ],
        [ 0.03207726, -0.01128823, -0.03940594],
        [ 0.03593286,  0.00417245,  0.01245406],
        [ 0.03686284, -0.0333414 , -0.00088217],
        [ 0.01968007,  0.02675997, -0.01278323],
        [ 0.05106105,  0.00542661,  0.02234752],
        [ 0.0218365 ,  0.00678694, -0.0329758 ],
        [-0.00059503,  0.01488227,  0.01848116],
        [ 0.02520229, -0.05689634,  0.01617914],
        [ 0.02217157, -0.0096695 , -0.00682362],
        [ 0.01130766, -0.02820945, -0.01314242],
        [-0.02321812, -0.00514337,  0.01609937],
        [ 0.03219966, -0.01274984, -0.00539225],
        [ 0.02241642,  0.03635795, -0.00359108],
        [ 0.016367  ,  0.0351348 ,  0.00614122],
        [-0.0276806 ,  0.03717545, -0.01038663],
        [ 0.0145419 , -0.00549463, -0.00282471],
        [-0.0364902 , -0.00287716,  0.00056543],
        [ 0.01910465, -0.0294238 , -0.02308549],
        [ 0.01663755,  0.04561475,  0.04118741],
        [ 0.01105849,  0.01560486, -0.03301357],
        [ 0.01047693,  0.04551702,  0.03776303],
        [ 0.01130631,  0.05163325, -0.00350297],
        [-0.06389773,  0.01007361, -0.03508611],
        [ 0.0308495 ,  0.00027086,  0.00360923],
        [ 0.0348151 , -0.04404767,  0.03173511],
        [-0.02405966, -0.03351495, -0.01024809],
        [ 0.02643059, -0.0287193 ,  0.00468121],
        [ 0.01912175, -0.03532287,  0.00931486],
        [ 0.02651113, -0.01979552, -0.01606957],
        [ 0.02929955, -0.00285982,  0.00703138],
        [-0.06484432, -0.01335897, -0.03216156],
        [-0.04022299, -0.05263656,  0.009311  ],
        [-0.05337869, -0.01346918,  0.0145786 ],
        [-0.03348142, -0.00186267,  0.00884088],
        [ 0.00668426,  0.00273427, -0.00224992],
        [ 0.02776045, -0.00216066, -0.01371621],
        [-0.02515659,  0.01451467, -0.01329626],
        [ 0.0216861 , -0.04305587, -0.01318057],
        [ 0.00416017,  0.02650947,  0.05195471],
        [-0.03742277, -0.00296079, -0.03715196],
        [-0.00913855, -0.0110325 ,  0.01491708],
        [ 0.02225312, -0.01350546,  0.00667084],
        [-0.01652805, -0.00100451, -0.01007608],
        [ 0.03141017, -0.00696115,  0.01931934],
        [ 0.03861434,  0.01449547, -0.00342058],
        [-0.01298207,  0.02853706, -0.00142001],
        [-0.02415835, -0.0008781 , -0.01136322],
        [ 0.0108345 , -0.01971089, -0.01336852],
        [ 0.03272221,  0.01524863,  0.01080296],
        [ 0.02371906,  0.01618476, -0.02559165],
        [ 0.00623116,  0.06261193,  0.01399687],
        [-0.03007973,  0.03630072,  0.009458  ],
        [ 0.00121658, -0.03084881,  0.01675261],
        [-0.04079645, -0.00305355, -0.01630775],
        [ 0.00133639, -0.0403649 , -0.0122901 ],
        [ 0.01609286, -0.01858252,  0.01655048],
        [-0.0379784 , -0.01471482,  0.00095278],
        [-0.01526452,  0.03042519, -0.00437549],
        [ 0.00099491, -0.06280597,  0.00989702],
        [ 0.01899159, -0.03753972, -0.03058725],
        [ 0.02905568, -0.04944663,  0.03953349],
        [-0.06188278,  0.02617978, -0.05193751],
        [ 0.02667367, -0.07366643,  0.0602624 ],
        [-0.02028434,  0.002268  , -0.01475613],
        [-0.03240722,  0.0184801 ,  0.0220475 ],
        [-0.00676903,  0.01610292, -0.0376305 ],
        [ 0.00027075, -0.01815558,  0.00095764],
        [-0.03181228, -0.01138223, -0.01112766],
        [ 0.08353256, -0.00531662,  0.09570256],
        [-0.02719685, -0.01212656, -0.06668041],
        [ 0.01314454,  0.03197736, -0.03764796],
        [ 0.01906369,  0.02602508, -0.00724198],
        [-0.01340752,  0.02362813,  0.05415871],
        [-0.02055424, -0.00458621, -0.00774949],
        [ 0.01421123, -0.05025844,  0.00068502],
        [ 0.03244635, -0.00306546, -0.03649294],
        [-0.01000641,  0.02344644,  0.02697884],
        [-0.00243039, -0.04153135, -0.01145939],
        [-0.00950665,  0.02036927, -0.00207437],
        [ 0.0315236 ,  0.0041688 ,  0.01560131],
        [ 0.04685416, -0.00238107,  0.00718689],
        [-0.01187091,  0.07182254, -0.018      ]
    ]

    angular_acc = [
        [ 6.96183917e-07, -7.58192905e-01,  2.42929991e-06],
        [-0.01235297,  1.16515089,  0.13145504],
        [-0.24457353,  0.57970402, -0.60761824],
        [-0.46576286, -0.77309944,  0.36846741],
        [ 1.04822089,  0.26456826, -0.79350361],
        [-1.4220633 , -0.81059663,  1.56314069],
        [ 0.31662855,  0.60456415, -0.66569331],
        [ 1.1026151 , -0.80836092, -1.16740253],
        [-0.42476994,  0.82605406, -0.03447353],
        [ 0.44886468, -0.41817201,  0.82868818],
        [ 0.48465263, -0.59469945, -1.14213873],
        [-1.00609054,  1.52452123,  2.72441599],
        [-0.56894737, -0.6740877 , -0.36530734],
        [ 0.98484565,  0.509225  ,  1.16284385],
        [-0.42411729, -0.7045791 , -3.61114142],
        [ 1.24591248,  0.70990062, -1.8388542 ],
        [-0.21860719, -1.06361841, -0.70829135],
        [-0.8671441 ,  0.60904648, -0.64978682],
        [-0.43889921, -0.50295323,  7.68246837],
        [ 0.13066505,  0.20573023, -0.30825225],
        [ 0.84602241,  0.77445494,  1.60764425],
        [-0.49808277, -0.94309255, -1.90652258],
        [-0.49230736,  3.16515314, -0.16413803],
        [ 0.53846393, -3.65641112, -6.63731642],
        [-0.61203419,  1.01896769,  5.67176116],
        [ 0.69885524,  0.38510659, -1.1929921 ],
        [-0.64537394, -1.32439604,  4.92756535],
        [ 1.35723403, -0.07490291, -1.09189508],
        [-0.48738032,  0.19550684, -0.60361783],
        [ 0.30477969, -0.59112327,  0.12688107],
        [-1.78001845,  2.77840863, -5.44398343],
        [ 0.40106098, -1.08691326,  0.40771578],
        [ 1.1430661 , -0.58827899,  0.32645351],
        [-0.50813145,  0.43316259, -3.26507809],
        [-0.00292397, -0.50855448,  1.85604254],
        [-0.12957254, -0.46599373,  1.34289668],
        [-0.00186283, -0.11578965,  0.66596016],
        [-0.52589994,  0.498213  ,  0.68944922],
        [ 0.73933422,  2.05641832, -4.34423337],
        [ 0.44775625, -1.98477064,  0.47303354],
        [-0.88889617, -0.94285625, -0.69690602],
        [-0.44540035,  1.33779923,  7.55762859],
        [ 1.364031  ,  1.87884813, -0.18305034],
        [-0.64279653, -4.24683134, -6.34586374],
        [ 0.20066443,  2.74992689,  5.94361634],
        [-0.50297535, -0.35793841, -1.53014017],
        [ 0.21987834, -1.84764086, -4.82214769],
        [ 0.85971272,  2.06137702,  6.01262918],
        [-0.14694908, -0.87426596, -5.43198289],
        [-0.56944491,  0.26211448,  0.04762795],
        [ 0.09202582,  0.33077605,  6.14147825],
        [ 0.39799371,  0.48350119,  0.654017  ],
        [-0.75041431, -1.61031018, -0.65118473],
        [ 0.46728017,  0.56112797, -6.06720557],
        [ 0.11558743,  0.2531226 ,  6.17884726],
        [ 0.26097799, -0.61532463, -1.40164068],
        [-0.37726401,  0.86755835, -0.6073279 ],
        [ 0.37859459,  0.79427165,  0.77396058],
        [-0.42322834, -1.67561405,  2.93114218],
        [-0.70966892,  1.03471785, -5.72307169],
        [ 0.33848812, -1.00778955, -3.25674483],
        [-0.16383711,  1.02996551,  5.59624288],
        [ 0.7158424 , -0.19740962, -4.76184982],
        [-0.42602252, -0.12255834, -0.28168557],
        [ 0.61932713, -0.41822492,  6.52961351],
        [-0.76439747,  0.1946552 , -4.92547408],
        [ 0.96715758,  1.20602669, -1.48226287],
        [ 0.02747176, -0.32276723, -0.28377785],
        [-2.29649001, -2.09250156,  2.04337808],
        [ 2.41338907,  2.47185163, -1.92906535],
        [-1.40521505, -1.45593172,  0.45259108],
        [ 0.51649217,  0.05195085,  7.70167448],
        [-0.39862017, -0.97604618, -8.60130041],
        [ 0.49841035,  1.80741957,  6.82604143],
        [-0.57756507, -0.64279122, -5.90237605],
        [ 1.95849273,  0.22539757,  5.67447628],
        [-3.46235266, -1.62427533,  0.25543511],
        [ 3.43699131,  1.46692339, -5.27659048],
        [ 0.39790096,  1.48126089,  6.0038325 ],
        [-1.26871486, -1.29627948,  1.26165811],
        [-1.23966918, -1.32084391, -8.38157843],
        [ 1.01407118,  1.60606706,  6.76178701],
        [-0.79834979, -1.40837615, -6.42137064],
        [ 1.08425415,  1.4582921 ,  6.07903399],
        [-1.16333476, -1.42424893, -5.62068913],
        [ 1.33356999,  1.40910591,  5.27075873]
    ]


    # Calculate mean and standard deviation
    mean_linear_acc = np.mean(linear_acc, axis=0)
    std_linear_acc = np.std(linear_acc, axis=0)

    mean_angular_acc = np.mean(angular_acc, axis=0)
    std_angular_acc = np.std(angular_acc, axis=0)

    # Print results
    print("Linear Acceleration (Mean ± Std Dev):")
    print(f"  X: {mean_linear_acc[0]:.3f} ± {std_linear_acc[0]:.3f}")
    print(f"  Y: {mean_linear_acc[1]:.3f} ± {std_linear_acc[1]:.3f}")
    print(f"  Z: {mean_linear_acc[2]:.3f} ± {std_linear_acc[2]:.3f}")

    print("\nAngular Acceleration (Mean ± Std Dev):")
    print(f"  X: {mean_angular_acc[0]:.3f} ± {std_angular_acc[0]:.3f}")
    print(f"  Y: {mean_angular_acc[1]:.3f} ± {std_angular_acc[1]:.3f}")
    print(f"  Z: {mean_angular_acc[2]:.3f} ± {std_angular_acc[2]:.3f}")

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(linear_acc)
    axs[0].set_title("Linear Acceleration")
    axs[0].legend(['X', 'Y', 'Z'])

    axs[1].plot(angular_acc)
    axs[1].set_title("Angular Acceleration")
    axs[1].legend(['X', 'Y', 'Z'])

    plt.tight_layout()
    plt.show()

    (mean_linear_acc, std_linear_acc, mean_angular_acc, std_angular_acc)


if __name__ == "__main__":
    # Example data generation
    np.random.seed(42)  # For reproducible results

    # Generate sample data
    timesteps = 100

    # Rewards: somewhat noisy with an increasing trend
    rewards = np.random.normal(0, 1, timesteps) + np.linspace(0, 2, timesteps)

    # Linear acceleration: 3D data with some patterns
    linear_accel = np.zeros((timesteps, 3))
    linear_accel[:, 0] = np.sin(np.linspace(0, 4*np.pi, timesteps)) * 2  # X component
    linear_accel[:, 1] = np.cos(np.linspace(0, 3*np.pi, timesteps)) * 1.5  # Y component
    linear_accel[:, 2] = np.sin(np.linspace(0, 2*np.pi, timesteps)) * 0.8  # Z component
    # Add some noise
    linear_accel += np.random.normal(0, 0.2, size=linear_accel.shape)

    # Angular acceleration: 3D data with different patterns
    angular_accel = np.zeros((timesteps, 3))
    angular_accel[:, 0] = np.sin(np.linspace(0, 6*np.pi, timesteps)) * 0.5  # X component
    angular_accel[:, 1] = np.sin(np.linspace(0, 5*np.pi, timesteps)) * 0.7  # Y component
    angular_accel[:, 2] = np.cos(np.linspace(0, 3*np.pi, timesteps)) * 0.9  # Z component
    # Add some noise
    angular_accel += np.random.normal(0, 0.1, size=angular_accel.shape)

    # Method 1: Plot all in one figure
    print("Plotting all data in one figure...")
    plot_simulation_history(rewards, linear_accel, angular_accel, window_size=10)

    # Method 2: Plot each separately
    #print("Plotting each dataset separately...")
    #plot_each_separately(rewards, linear_accel, angular_accel, window_size=10)
    print("Done!")