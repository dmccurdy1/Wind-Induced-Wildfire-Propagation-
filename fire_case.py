import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
import pickle

class Inferno(): # This class propegates fire on a field using random values and a constant probability threshold 

    def __init__(self, case = None, p_grid = None, direction = None, wind_mag = 1): # initializing attributes for different cases

        if case == 1:
            self.p_grid = None
            self.case = 1
            self.p = 0.5
            self.time_step = 1
            self.grid_dim = 3
            self.initial_position = [(1,1)]
        elif case == 2:
            self.p_grid = None
            self.case = 2
            self.p = 0.3
            self.grid_dim = 6
            self.time_step = 1
            self.initial_position = [(2,2),(3,2),(3,3),(2,3)]
        else:
            self.p_grid = None
            self.case = None
            self.grid_dim =11
            self.time_step = 50
            self.initial_position = [(9,5)]#[(int(self.grid_dim/2),int(self.grid_dim/2))]
            if p_grid == "On":
                self.p_grid = None
                self.p = np.linspace(0.1,0.5,self.grid_dim)
                self.p = np.tile(self.p, (self.grid_dim,1))
                self.p = self.p.T
            elif p_grid == "Wind":
                self.p_grid = "Wind"
                if direction == "North" or direction == None:
                    self.wind_grid = [[(wind_mag, (1*np.pi/2)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == "South":
                    self.wind_grid = [[(wind_mag, (3*np.pi/2)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == "East":
                    self.wind_grid = [[(wind_mag, (0)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == "West":
                    self.wind_grid = [[(wind_mag, (np.pi)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == 'North East':
                    self.wind_grid = [[(wind_mag, (np.pi/8)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == 'North West':
                    self.wind_grid = [[(wind_mag, (5*np.pi/8)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == 'South West':
                    self.wind_grid = [[(wind_mag, (9*np.pi/8)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == 'South East':
                    self.wind_grid = [[(wind_mag, (13*np.pi/8)) for i in range(self.grid_dim)] for i in range(self.grid_dim)]
                elif direction == 'Vortex':                    

                    # Define the dimensions of the list
                    rows = self.grid_dim
                    cols = self.grid_dim

                    # Calculate the increment value
                    increment = 1 * math.pi / (rows - 1)

                    # Generate the 2D list of tuples
                    result = [[(wind_mag, 0)] * cols for _ in range(rows)]

                    # Fill the list with the desired values
                    for i in range(rows):
                        for j in range(cols):
                            result[i][j] = (wind_mag, i * increment)

                    self.wind_grid = result

                self.wind_mag = wind_mag
                self.p = 1.
            else:
                self.p = 0.1

    def make_square_grid(self): # outputs a grid of zeros with an value of 1 at the initial position

        grid = np.matrix(np.zeros((self.grid_dim,self.grid_dim)))

        for i in range(0,len(self.initial_position)):

            grid[self.initial_position[i]] = grid[self.initial_position[i]] + 1

        return grid

    def find_adjacents(self, grid, count, new_sqaures): # outputs the index of all adjacent cells to value of 1 cells

        if count == 0:
            locations_of_ones = np.argwhere(grid != 0)
        else:
            locations_of_ones = np.argwhere(new_sqaures != 0)
        adjacents = list()

        for i in locations_of_ones: # finding all adjacent cells and adding to list

            #start = time.time()

            up = (i[0] - 1, i[1])
            down = (i[0] + 1, i[1])

            left = (i[0], i[1] - 1)
            right = (i[0], i[1] + 1)
            
            up_left = (i[0] - 1 , i[1] - 1)
            up_right = (i[0] - 1 , i[1] + 1)

            down_left = (i[0] + 1, i[1] - 1)
            down_right = (i[0] + 1, i[1] + 1)

            #end = time.time()

            #print('>time inside up/down...:', (end-start))

            #start = time.time()
            
            if up[0] < self.grid_dim and up[0] >= 0 and up[1] < self.grid_dim and up[1] >= 0:
                adjacents.append(up)
            if down[0] < self.grid_dim and down[0] >= 0 and down[1] < self.grid_dim and down[1] >= 0:
                adjacents.append(down)
            if left[0] < self.grid_dim and left[0] >= 0 and left[1] < self.grid_dim and left[1] >= 0:
                adjacents.append(left)
            if right[0] < self.grid_dim and right[0] >= 0 and right[1] < self.grid_dim and right[1] >= 0:
                adjacents.append(right)
            if up_left[0] < self.grid_dim and up_left[0] >= 0 and up_left[1] < self.grid_dim and up_left[1] >= 0:
                adjacents.append(up_left)
            if up_right[0] < self.grid_dim and up_right[0] >= 0 and up_right[1] < self.grid_dim and up_right[1] >= 0:
                adjacents.append(up_right)
            if down_left[0] < self.grid_dim and down_left[0] >= 0 and down_left[1] < self.grid_dim and down_left[1] >= 0:
                adjacents.append(down_left)
            if down_right[0] < self.grid_dim and down_right[0] >= 0 and down_right[1] < self.grid_dim and down_right[1] >= 0:
                adjacents.append(down_right)

            #end = time.time()

            #print('>time inside if statements:', (end-start))

            for f in range(0,len(locations_of_ones)): # removing cells that were already valued 1

                if tuple(locations_of_ones[f]) in adjacents:
                    adjacents.remove(tuple(locations_of_ones[f]))
                else:
                    pass 

        return adjacents

    def make_random_sqaure_grid(self): # makes a matrix with same dimensions and grid and assigns random value between 0-1 to every cell

        random_matrix = np.random.rand(self.grid_dim,self.grid_dim)

        return random_matrix

    def propegate(self, grid, adjacent_cells, random_values = None, wind_p_grid = None, wind_cells = None): # creates new 1 valued cells if probability condition is met, returns updated matrix
        
        with open('initial_grid.pkl', 'wb') as f:
            pickle.dump(grid, f)

        downstream_cells, turbulent_cells = wind_cells

        downstream_prob, turbulent_prob = self.mag_to_p(self.wind_mag)
        base_prob = 0.3

        #breakpoint()

        for i in downstream_cells:

            random_values = self.make_random_sqaure_grid()

            if random_values[i] < downstream_prob:

                grid[i] = grid[i] + 1

                # if i in turbulent_cells:
                #     turbulent_cells.remove(i)
                # if i in adjacent_cells:
                #     adjacent_cells.remove(i)

        for i in turbulent_cells:

            random_values = self.make_random_sqaure_grid()

            if random_values[i] < turbulent_prob:

                grid[i] = grid[i] + 1

                # if i in adjacent_cells:
                #     adjacent_cells.remove(i)
                # if i in downstream_cells:
                #     downstream_cells.remove(i)

        for i in adjacent_cells:

            random_values = self.make_random_sqaure_grid()

            if random_values[i] < base_prob:

                grid[i] = grid[i] + 1


                # if i in downstream_cells:
                #     downstream_cells.remove(i)
                # if i in turbulent_cells:
                #     turbulent_cells.remove(i)

        with open('initial_grid.pkl', 'rb') as f:
            initial_grid = pickle.load(f)

        for i in range(self.grid_dim):
            for f in range(self.grid_dim):
                if grid[i,f] > 1:
                    grid[i,f] = 1

        new_squares = grid - initial_grid
        
        return grid, new_squares

    def wildfire(self): # iterates through grid and propegates fire returns list of each grid iteration
        
        grid = self.make_square_grid()

        grid_iterations = [grid.tolist()]

        count = 0
        new_squares = []

        for i in range(0,self.time_step):

            random_grid = self.make_random_sqaure_grid()

            #start = time.time()

            adjacent_cells = self.find_adjacents(grid, count, new_squares)
            count += 1

            #end = time.time()

            #print('time inside adjacent cells:', (end-start))

            if self.p_grid == "Wind":

                #start = time.time()

                wind_forced_cells = self.wind_spread(grid, self.wind_grid)

                #end = time.time()

                #print('time inside wind spread:', (end-start))

                #start = time.time()

                grid, new_squares = self.propegate(grid, adjacent_cells, wind_cells=wind_forced_cells)
                
                #end = time.time()

                #print('time inside progagate:', (end-start))
            
            else:

                grid = self.propegate(grid, adjacent_cells, random_grid)

            grid_iterations.append(grid.tolist())

        return grid_iterations

    def animate_fire(self, average = None): # creates iteratable frames and shows animation

        if average == 'On':
            fire = self.average_fire()
        else:
            fire = self.wildfire()

        fig, axs = plt.subplots()

        def frame_function(frame):
            axs.clear()
            axs.imshow(fire[frame], cmap='hot', interpolation='nearest')

        animatation = animation.FuncAnimation(fig, frame_function, frames=len(fire), interval=100)

        if self.case == 1 or self.case == 2:

            print('p value is {}'.format(self.p))
            print(fire[-1])

        plt.show()

    def average_fire(self,iterations = 100): # averages time steps of many different wildfire calls

        fire_runs = []

        for i in range(0, iterations):

            fire_in_loop = self.wildfire()

            fire_runs.append(fire_in_loop)

        mean_fire = np.mean(fire_runs,axis = 0)

        if self.case == 1 or self.case == 2:

            print('------case {}--------'.format(self.case))
            print('')
            print('p value is {}'.format(self.p))
            print('')
            print(mean_fire[-1])
            print('')
            print('--------------------')

        return mean_fire

    def mag_to_p(self, wind_mag): # converts wind magnitude to a probability of propegation

        # if wind_mag > 25:
        #         p_in_cell = 1
        if wind_mag < 0:
            raise ValueError('Only positive magnitude values are accepted')
        else:
            p_in_cell = np.tanh(0.1*wind_mag) #(np.log(2*wind_mag+1)) / 4
            turb_prob = p_in_cell / 4
        
        return p_in_cell, turb_prob

    def wind_spread(self, grid, wind_grid): # takes fire matrix and updates probability field based on wind streams

        # Assumptions: wind direction is constant, wind magnatude is unifrom, (laminar, imcopmressable)
        # proability relation to wind speed is logarithmic
        # wind can propegate in eight discrete directions

        fire_locations = np.argwhere(grid != 0)

        wind_p_grid = [[0.1] * self.grid_dim for i in range(self.grid_dim)]

        downstream_cells = []
        turbulent_cells = []

        for i in fire_locations:

            wind_dir_at_fire = wind_grid[i[0]][i[1]][1]
            wind_mag_at_fire = wind_grid[i[0]][i[1]][0]
            p_from_wind_mag, turb_prob = self.mag_to_p(wind_mag_at_fire)

            turb_p = p_from_wind_mag/4


            if wind_dir_at_fire >= 0 and wind_dir_at_fire < (np.pi / 8): # right

                if i[1] + 1 >= self.grid_dim:
                    pass
                
                else:
                    wind_p_grid[i[0]][i[1] + 1] = wind_p_grid[i[0]][i[1] + 1] + p_from_wind_mag
                
                    downstream_cell = (i[0], i[1] + 1)
                    downstream_cells.append(downstream_cell)

                    # peripheral spread
                    if i[0] - 1 < 0 or i[0] + 1 > self.grid_dim or i[1] + 1 > self.grid_dim:
                        pass
                    else:
                        wind_p_grid[i[0] - 1][i[1] + 1] = wind_p_grid[i[0] - 1][i[1] + 1] + (turb_p)
                        wind_p_grid[i[0] + 1][i[1] + 1] = wind_p_grid[i[0] + 1][i[1] + 1] + (turb_p)

                        turbulent_cell_1 = (i[0] - 1, i[1] + 1)
                        turbulent_cell_2 = (i[0] + 1, i[1] + 1)

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (np.pi / 8) and wind_dir_at_fire < (3*np.pi/8): # up right

                if i[0] - 1 < 0 or i[1] + 1 >= self.grid_dim:
                    pass

                else:
                    wind_p_grid[i[0] - 1][i[1] + 1] = wind_p_grid[i[0] - 1][i[1] + 1] + p_from_wind_mag
                    downstream_cell = (i[0] - 1, i[1] + 1)
                    downstream_cells.append(downstream_cell)

                    # peripheral spread
                    if i[0] - 1 < 0 or i[1] + 1 > self.grid_dim:
                        pass
                    else:
                        wind_p_grid[i[0] - 1][i[1]] = wind_p_grid[i[0] - 1][i[1]] + (turb_p)
                        wind_p_grid[i[0]][i[1] + 1] = wind_p_grid[i[0]][i[1] + 1] + (turb_p)

                        turbulent_cell_1 = (i[0] - 1, i[1])
                        turbulent_cell_2 = (i[0], i[1] + 1)

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (3*np.pi/8) and wind_dir_at_fire < (5*np.pi/8): # up

                if i[0] - 1 < 0:
                    pass

                else:
                    wind_p_grid[i[0] - 1][i[1]] = wind_p_grid[i[0] - 1][i[1]] + p_from_wind_mag
                    downstream_cell = (i[0] - 1, i[1])
                    downstream_cells.append(downstream_cell)

                    if i[0] - 1 < 0 or i[1] - 1 < 0 or i[1] + 1 >= self.grid_dim:
                        pass
                    else:
                        wind_p_grid[i[0] - 1][i[1] - 1] = wind_p_grid[i[0] - 1][i[1] - 1] + (turb_p)
                        wind_p_grid[i[0] - 1][i[1] + 1] = wind_p_grid[i[0] - 1][i[1] + 1] + (turb_p)

                        turbulent_cell_1 = (i[0] - 1, i[1] - 1)
                        turbulent_cell_2 = (i[0] - 1, i[1] + 1)

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (5*np.pi/8) and wind_dir_at_fire < (7*np.pi/8): # up left

                if i[0] - 1 < 0 or i[1] - 1 < 0:
                    pass

                else:  
                    wind_p_grid[i[0] - 1][i[1] - 1] = wind_p_grid[i[0] - 1][i[1] - 1] + p_from_wind_mag
                    downstream_cell = (i[0] - 1,i[1] - 1)
                    downstream_cells.append(downstream_cell)

                    # peripheral spread
                    if i[0] - 1 < 0 or i[1] - 1 < 0:
                        pass
                    else:
                        wind_p_grid[i[0]][i[1] - 1] = wind_p_grid[i[0]][i[1] - 1] + (turb_p)
                        wind_p_grid[i[0] - 1][i[1]] = wind_p_grid[i[0] - 1][i[1]] + (turb_p)

                        turbulent_cell_1 = (i[0], i[1] - 1)
                        turbulent_cell_2 = (i[0] - 1, i[1])

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (7*np.pi/8) and wind_dir_at_fire < (9*np.pi/8): # left

                if i[1] - 1 < 0:
                    pass

                else:
                    wind_p_grid[i[0]][i[1] - 1] = wind_p_grid[i[0]][i[1] - 1] + p_from_wind_mag
                    downstream_cell = (i[0],i[1] - 1)
                    downstream_cells.append(downstream_cell)

                    # peripheral spread
                    if i[0] + 1 >= self.grid_dim or i[1] - 1 < 0:
                        pass
                    else:    
                        wind_p_grid[i[0] - 1][i[1] - 1] = wind_p_grid[i[0] - 1][i[1] - 1] + (turb_p)
                        wind_p_grid[i[0] + 1][i[1] - 1] = wind_p_grid[i[0] + 1][i[1] - 1] + (turb_p)

                        turbulent_cell_1 = (i[0] - 1, i[1] - 1)
                        turbulent_cell_2 = (i[0] + 1, i[1] - 1)

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (9*np.pi/8) and wind_dir_at_fire < (11*np.pi/8): # down left

                if i[0] + 1 >= self.grid_dim or i[1] - 1 < 0:
                    pass

                else:
                    wind_p_grid[i[0] + 1][i[1] - 1] = wind_p_grid[i[0] + 1][i[1] - 1] + p_from_wind_mag
                    downstream_cell = (i[0] + 1,i[1] - 1)
                    downstream_cells.append(downstream_cell)

                    # peripheral spread
                    if i[0] + 1 > self.grid_dim or i[1] - 1 < 0:
                        pass
                    else:    
                        wind_p_grid[i[0]][i[1] - 1] = wind_p_grid[i[0]][i[1] - 1] + (turb_p)
                        wind_p_grid[i[0] + 1][i[1]] = wind_p_grid[i[0] + 1][i[1]] + (turb_p)

                        turbulent_cell_1 = (i[0], i[1] - 1)
                        turbulent_cell_2 = (i[0] + 1, i[1])

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (11*np.pi/8) and wind_dir_at_fire < (13*np.pi/8): # down

                if i[0] + 1 >= self.grid_dim:
                    pass

                else:
                    wind_p_grid[i[0] + 1][i[1]] = wind_p_grid[i[0] + 1][i[1]] + p_from_wind_mag
                    downstream_cell = (i[0] + 1,i[1])
                    downstream_cells.append(downstream_cell)

                    # peripheral spread
                    if i[0] + 1 >= self.grid_dim or i[1] - 1 < 0 or i[1] + 1 >= self.grid_dim:
                        pass
                    else:
                        wind_p_grid[i[0] + 1][i[1] - 1] = wind_p_grid[i[0] + 1][i[1] - 1] + (turb_p)
                        wind_p_grid[i[0] + 1][i[1] + 1] = wind_p_grid[i[0] + 1][i[1] + 1] + (turb_p)

                        turbulent_cell_1 = (i[0] + 1, i[1] - 1)
                        turbulent_cell_2 = (i[0] + 1, i[1] + 1)

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (13*np.pi/8) and wind_dir_at_fire < (15*np.pi/8): # down right

                if i[0] + 1 >= self.grid_dim or i[1] + 1 >= self.grid_dim:
                    pass
                else:
                    wind_p_grid[i[0] + 1][i[1] + 1] = wind_p_grid[i[0] + 1][i[1] + 1] + p_from_wind_mag
                    downstream_cell = (i[0] + 1,i[1] + 1)
                    downstream_cells.append(downstream_cell)


                    # peripheral spread
                    if i[0] + 1 > self.grid_dim or i[1] + 1 > self.grid_dim:
                        pass
                    else:
                        wind_p_grid[i[0] + 1][i[1]] = wind_p_grid[i[0] + 1][i[1]] + (turb_p)
                        wind_p_grid[i[0]][i[1] + 1] = wind_p_grid[i[0]][i[1] + 1] + (turb_p)

                        turbulent_cell_1 = (i[0] + 1, i[1])
                        turbulent_cell_2 = (i[0], i[1] + 1)

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            elif wind_dir_at_fire >= (15*np.pi/8) and wind_dir_at_fire <= (2*np.pi): #up

                if i[1] + 1 >= self.grid_dim:
                    pass
                else:

                    wind_p_grid[i[0]][i[1] + 1] = wind_p_grid[i[0]][i[1] + 1] + p_from_wind_mag
                    downstream_cell = (i[0],i[1] + 1)
                    downstream_cells.append(downstream_cell)

                    # peripheral spread
                    if i[0] - 1 < 0 or i[0] + 1 >= self.grid_dim or i[1] + 1 >= self.grid_dim:
                        pass
                    else:
                        wind_p_grid[i[0] - 1][i[1] + 1] = wind_p_grid[i[0] - 1][i[1] + 1] + (turb_p)
                        wind_p_grid[i[0] + 1][i[1] + 1] = wind_p_grid[i[0] + 1][i[1] + 1] + (turb_p)

                        turbulent_cell_1 = (i[0] - 1, i[1] + 1)
                        turbulent_cell_2 = (i[0] + 1, i[1] + 1)

                        turbulent_cells.append(turbulent_cell_1)
                        turbulent_cells.append(turbulent_cell_2)

            else:

                raise ValueError('Direction needs to be between 0-2pi')
        
        wind_p_grid = np.array(wind_p_grid)

        for f in range(0,len(fire_locations)): # removing cells that were already valued 1

                if tuple(fire_locations[f]) in downstream_cells:
                    downstream_cells.remove(tuple(fire_locations[f]))
                else:
                    pass 
        for f in range(0,len(fire_locations)): # removing cells that were already valued 1

                if tuple(fire_locations[f]) in turbulent_cells:
                    turbulent_cells.remove(tuple(fire_locations[f]))
                else:
                    pass 

        wind_forced_cells = downstream_cells, turbulent_cells

        return wind_forced_cells

    def solve_vector_field(self): # Uses wind matrix to create a U & V vector field for plots

        rows = self.grid_dim
        cols = self.grid_dim

        self.U_comp = []
        self.V_comp = []
        for i in range(rows):
            for j in range(cols):
                u_comp = np.cos(self.wind_grid[i][j][1])
                v_comp = np.sin(self.wind_grid[i][j][1])

                self.U_comp.append(u_comp)
                self.V_comp.append(v_comp)
        
    def plot_vectors(self): # plots the U & V wind field

        fig, axs = plt.subplots()

        x = np.linspace(0,1,self.grid_dim)
        y = -np.linspace(0,1,self.grid_dim)

        X, Y = np.meshgrid(x,y)

        self.solve_vector_field()

        U = self.U_comp
        V = self.V_comp

        axs.quiver(X, Y, U, V, color='g') 

        plt.show()

    def plot_fire_snapshot(self, average = "On", no_wind_compare = None): # Contour plots of fire at different times

        if no_wind_compare != None:

            # Generate Fire

            if average == 'On':
                fire_with_wind = Inferno(case = 3, p_grid = 'Wind', direction = 'North', wind_mag=1).average_fire()
                fire_no_wind = Inferno(case = 3, p_grid = 'Wind', direction = 'North', wind_mag=0).average_fire()
            else:
                fire_with_wind = self.wildfire()
                fire_no_wind = Inferno(case = 3).wildfire()

            # Set Up Plotting Grid

            x = np.linspace(0,1,self.grid_dim)
            y = -np.linspace(0,1,self.grid_dim)

            X, Y = np.meshgrid(x,y)

            # Solve Vector Field

            self.solve_vector_field()

            U = self.U_comp
            V = self.V_comp

           # breakpoint()

            # Plot !

            fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (10,10))
        

            fig.suptitle('Effects of Wind on Fire Propegation In Model')

            axs[0,0].contourf(X,Y,fire_with_wind[0][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))
            axs[0,1].contourf(X,Y,fire_with_wind[int(self.time_step/2)][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))
            contourf = axs[0,2].contourf(X,Y,fire_with_wind[self.time_step][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))

            axs[0,0].quiver(X, Y, U, V, color='white')
            axs[0,1].quiver(X, Y, U, V, color='white')
            axs[0,2].quiver(X, Y, U, V, color='white')

            axs[0,0].plot(0.5,-0.3, 'ro', color = 'blue')
            axs[0,1].plot(0.5,-0.3, 'ro', color = 'blue')
            axs[0,2].plot(0.5,-0.3, 'ro', color = 'blue', label = 'home')

            axs[0,0].set_ylabel('Y')
        
            axs[0,0].set_title('T = 0')
            axs[0,1].set_title('With Wind \n T = 0.5')
            axs[0,2].set_title('T = 1')

            axs[0,0].set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
            axs[0,1].set_yticklabels([])
            axs[0,2].set_yticklabels([])

            axs[0,0].set_xticklabels([])
            axs[0,1].set_xticklabels([])
            axs[0,2].set_xticklabels([])
        
            # ---------------------------------------------------------------- #

            axs[1,0].contourf(X,Y,fire_no_wind[0][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))
            axs[1,1].contourf(X,Y,fire_no_wind[int(self.time_step/2)][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))
            contourf = axs[1,2].contourf(X,Y,fire_no_wind[self.time_step][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))

            axs[1,0].plot(0.5,-0.3, 'ro', color = 'blue')
            axs[1,1].plot(0.5,-0.3, 'ro', color = 'blue')
            axs[1,2].plot(0.5,-0.3, 'ro', color = 'blue', label = 'home')

            axs[1,0].set_ylabel('Y')
            axs[1,0].set_xlabel('X')
            axs[1,1].set_xlabel('X')
            axs[1,2].set_xlabel('X')

            axs[1,1].set_title('No Wind')
        
            axs[1,0].set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
            axs[1,1].set_yticklabels([])
            axs[1,2].set_yticklabels([])

            axs[1,0].set_xticklabels([0,0.2,0.4,0.6,0.8,1])
            axs[1,1].set_xticklabels([0,0.2,0.4,0.6,0.8,1])
            axs[1,2].set_xticklabels([0,0.2,0.4,0.6,0.8,1])

            fig.subplots_adjust(right=0.8)
            axs[0,2].legend(loc = 'upper right')
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(contourf, cax=cbar_ax, label = 'Fire Probability')


            # ---------------------------------------------------------------- #

            # difference_t0 = fire_no_wind[0][:][:] - fire_with_wind[0][:][:]
            # difference_t05 = fire_no_wind[int(self.time_step/2)][:][:] - fire_with_wind[int(self.time_step/2)][:][:]
            # difference_t1 = fire_no_wind[int(self.time_step)][:][:] - fire_with_wind[int(self.time_step)][:][:]


            # axs[2,0].contourf(X,Y,difference_t0, np.arange(-1,1,0.01),cmap=plt.get_cmap('bwr'))
            # axs[2,1].contourf(X,Y,difference_t05, np.arange(-1,1,0.01),cmap=plt.get_cmap('bwr'))
            # contourf = axs[2,2].contourf(X,Y,difference_t1, np.arange(-1,1,0.01),cmap=plt.get_cmap('bwr'))



            # fig.tight_layout()

            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # fig.colorbar(contourf, cax=cbar_ax, label = 'Fire Probability')

            # axs[0,2].legend(loc = 'upper right')



    
            plt.savefig('wind_v_no_wind_prop.png')
            #plt.show()

        else:

            if average == 'On':
                fire = self.average_fire()
            else:
                fire = self.wildfire()

            x = np.linspace(0,1,self.grid_dim)
            y = -np.linspace(0,1,self.grid_dim)

            X, Y = np.meshgrid(x,y)

            # Solve Vector Field

            self.solve_vector_field()

            U = self.U_comp
            V = self.V_comp

            # Plot !

            fig, axs = plt.subplots(ncols = 3, figsize = (10,4))

            fig.suptitle('Wind Vector Field Superimposed onto Fire Propegation over Time')

            axs[0].contourf(X,Y,fire[0][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))
            axs[1].contourf(X,Y,fire[int(self.time_step/2)][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))
            contourf = axs[2].contourf(X,Y,fire[self.time_step][:][:], np.arange(0,1,0.001),cmap=plt.get_cmap('hot'))
            
            plt.colorbar(contourf, ax=axs, label = 'Fire Probability')

            axs[0].quiver(X, Y, U, V, color='white')
            axs[1].quiver(X, Y, U, V, color='white')
            axs[2].quiver(X, Y, U, V, color='white')

            axs[0].plot(0.5,-0.3, 'ro', color = 'blue')
            axs[1].plot(0.5,-0.3, 'ro', color = 'blue')
            axs[2].plot(0.5,-0.3, 'ro', color = 'blue', label = 'home')

            axs[0].set_ylabel('Y')
            axs[0].set_xlabel('X')
            axs[1].set_xlabel('X')
            axs[2].set_xlabel('X')

            axs[0].set_title('t = 0')
            axs[1].set_title('t = 0.5')
            axs[2].set_title('t = 1')

            axs[0].set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
            axs[1].set_yticklabels([])
            axs[2].set_yticklabels([])

            axs[0].set_xticklabels([0,0.2,0.4,0.6,0.8,1])
            axs[1].set_xticklabels([0,0.2,0.4,0.6,0.8,1])
            axs[2].set_xticklabels([0,0.2,0.4,0.6,0.8,1])

            # axs[0].legend()
            # axs[1].legend()
            # axs[2].legend()

            plt.legend()

            #plt.savefig('wind_field_and_fire_prop.eps')
            plt.show()

    def propagation_rate(self): # solves for upstream and downstream propagation rates

        probability_threshold = 0.7

        fire_data = self.average_fire()
        intial_column = self.initial_position[0][1]
        intial_row = self.initial_position[0][0]

        downstream_rate_of_propegation = []
        upstream_rate_of_propegation = []

        locations_of_downstream_edge = []

        for i in range(1,self.time_step): # Solving Downstreama and Upstream Propagations

            fire_at_time = fire_data[i]

            column_of_propegation = fire_at_time[:,intial_column]

            locations_of_high_prob = np.argwhere(column_of_propegation >= probability_threshold)

            loc_of_farthest_from_source_downstream = np.min(locations_of_high_prob)
            loc_of_farthest_from_source_upstream = np.max(locations_of_high_prob)

            distance_from_source_downstream = intial_row - loc_of_farthest_from_source_downstream
            distance_from_source_upstream = loc_of_farthest_from_source_upstream - intial_row 

            downstream_rate = distance_from_source_downstream / i
            upstream_rate = distance_from_source_upstream / i

            downstream_rate_of_propegation.append(downstream_rate)
            upstream_rate_of_propegation.append(upstream_rate)

            locations_of_downstream_edge.append(loc_of_farthest_from_source_downstream)

        downstream_velocity = abs(np.gradient(locations_of_downstream_edge))

        rates = downstream_rate_of_propegation, upstream_rate_of_propegation           
        
        return downstream_velocity

    def plot_rates(self): # Plots rates

        downstream_velocity_0 = Inferno(case = 3, p_grid = 'Wind', direction = 'North', wind_mag=0).propagation_rate()
        downstream_velocity_1 = Inferno(case = 3, p_grid = 'Wind', direction = 'North', wind_mag=1).propagation_rate()
        downstream_velocity_5 = Inferno(case = 3, p_grid = 'Wind', direction = 'North', wind_mag=5).propagation_rate()

        np.save('downstream_rate_0.npy', downstream_velocity_0)
        np.save('downstream_rate_1.npy', downstream_velocity_1)
        np.save('downstream_rate_5.npy', downstream_velocity_5)

        downstream_rate_1 = np.load('downstream_rate_1.npy', allow_pickle=True)
        downstream_rate_5 = np.load('downstream_rate_5.npy', allow_pickle=True)
        downstream_rate_0 = np.load('downstream_rate_0.npy', allow_pickle=True)

        average_5 = np.mean(downstream_rate_5)
        average_1 = np.mean(downstream_rate_1)
        average_0 = np.mean(downstream_rate_0)

        time_ax = np.linspace(1,self.time_step,(self.time_step - 1))

        fig, axs = plt.subplots(nrows = 1)

        axs.plot(time_ax[0:10],downstream_rate_1[0:10], label = '1 m/s')
        #axs[1].plot(time_ax,upstream_rate_1, label = '1 m/s')

        axs.plot(time_ax[0:10],downstream_rate_5[0:10], label = '5 m/s')
        #axs[1].plot(time_ax,upstream_rate_5, label = '5 m/s')

        #axs[0].plot(time_ax,downstream_rate_10, label = '10 m/s')
        #axs[1].plot(time_ax,upstream_rate_10, label = '10 m/s')

        axs.plot(time_ax[0:10],downstream_rate_0[0:10], label = '0 m/s')
        #axs[1].plot(time_ax,upstream_rate_nowind, label = 'no wind')

        axs.set_xlabel('time (iter)')
        axs.set_ylabel('propagation rate (cell/iteration)')
        axs.set_title('Average Rates: 5m/s: {:.2f}, 1m/s: {:.2f}, 0m/s: {:.2f}'.format(average_5,average_1,average_0))
        fig.suptitle('Propagation Rate v Time for No Wind and Downstream of Wind')

        axs.legend()
        #axs.legend()

        plt.savefig("propagation_rate.png")

    def rate_analysis(self): # determines downstream rate of fire spread

        downstream_rate_1 = np.load('downstream_1.npy', allow_pickle=True)
        downstream_rate_5 = np.load('downstream_5.npy', allow_pickle=True)
        downstream_rate_nowind = np.load('downstream_nowind.npy', allow_pickle=True)

        mean_rate_1 = np.mean(downstream_rate_1)
        mean_rate_5 = np.mean(downstream_rate_5)
        mean_rate_nowind = np.mean(downstream_rate_nowind)

        percent_diff_nowindto1 = (abs(mean_rate_nowind - mean_rate_1) / ((mean_rate_nowind + mean_rate_1)/2)) * 100
        percent_diff_1to5 = (abs(mean_rate_1 - mean_rate_5) / ((mean_rate_1 + mean_rate_5)/2)) * 100

        breakpoint()

if __name__ == '__main__':
    
    # Can input 8 unique directions of wind: North, South, East, West, North East, North West, South East, and South West

    Inferno().plot_rates()

    
    #Inferno(case = 3, p_grid = 'Wind', direction = 'North', wind_mag=1).animate_fire(average = 'On')


    #Inferno(case = 3, p_grid = 'Wind', direction = 'North').plot_fire_snapshot(no_wind_compare = 'On')
    #Inferno(case = 3, p_grid = 'Wind', direction = 'North').plot_fire_snapshot(wind)
    # Inferno(case = 3, p_grid = 'Wind', direction = 'Vortex').plot_vectors()