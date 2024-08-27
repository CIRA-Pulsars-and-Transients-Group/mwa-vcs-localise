#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

def data_reader(data_directory):
    tabp_file = os.path.join(data_directory,'tabp_look.npy')
    grid_file = os.path.join(data_directory,'grid.npz')
    
    tabp_look = np.load(tabp_file)
    print("Selecing the first frequency and ignoring all other frequencies")
    tabp_look = tabp_look[:,0,:,:]
    grid_ra = np.load(grid_file)['arr_0']
    grid_dec = np.load(grid_file)['arr_1']
    #product = np.sum(tabp_look.mean(axis=1), axis=0)
    print('tabp dimensions:', tabp_look.shape)
    #print('product dimensions:', product.shape)
    print('RA/Dec grid dimensions:', grid_ra.shape, grid_dec.shape)
    
    return tabp_look, grid_ra, grid_dec


def beam_plot(beam_cen_coords, tabp, grid_ra, grid_dec, label, contours=True):
    tabp_sum = np.sum(tabp, axis=0)

    map_extent = [grid_ra.min(),
                  grid_ra.max(),
                  grid_dec.min(),
                  grid_dec.max()]
    
    aspect = 'equal'    

    cmap = cms.get_sub_cmap(cms.cosmic,0.1,0.9)
    cmap.set_bad('red')
    contour_cmap = cms.get_sub_cmap(cms.cosmic_r,0.1,0.9)
    cmapnorm_sum = colors.Normalize(vmin=1e-5, vmax=0.1, clip=True)
    cmapnorm_indiv = colors.Normalize(vmin=1e-5, vmax=0.05, clip=True)
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1,1,1)
    ax1_img = ax1.imshow(tabp_sum,
                         aspect=aspect, extent=map_extent, cmap=cmap, norm=cmapnorm_sum)

    ax1.plot(beam_cen_coords.ra.deg,beam_cen_coords.dec.deg,'Dy',mec='k', ms=5, label='Beam centers')

    if contours:
        for ls, look in enumerate(tabp):
            ct = ax1.contour(look, origin='image', extent=map_extent, cmap=contour_cmap, norm=cmapnorm_indiv, linewidths=0.5)
    
    ax1.legend(fontsize=18, loc=2)
    ax1.set_xlabel('R.A. (ICRS)',fontsize=18,ha='center')
    ax1.set_ylabel('Dec. (ICRS)',fontsize=18,ha='center')
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.tick_params(axis='both', which='major', length=9)
    ax1.tick_params(axis='both', which='minor', length=4.5)
    ax1.tick_params(axis='both', which='both', direction='out', right=True, top=True)

    cbar = fig.colorbar(ax1_img, ax=fig.axes, shrink=1, orientation='horizontal',location='top', aspect=30, pad=0.02)
    cbar.ax.set_title(label,fontsize=18,ha='center')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(direction='in',length=5,bottom=True,top=True)
    cbar.ax.xaxis.set_tick_params(labelsize=18)
    return fig, ct


