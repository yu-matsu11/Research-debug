
# Hz = np.exp(1j * 2* np.pi * z * np.sqrt(term))
# G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_pad)))
# G_recon = G * Hz
# print(np.mean(G_recon))
# g_recon_pad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_recon)))

# X_start, Y_start = Nx // 2, Ny // 2
# g_recon = g_recon_pad[Y_start : Y_start+Ny, X_start : X_start+Nx]

# g_recon_abs = np.abs(g_recon)
# if g_recon_abs.max() > 0:
#     g_view = (255 * (g_recon_abs / g_recon_abs.max())).astype(np.uint8)
# else:
#     g_view = np.zeros_like(g_recon_abs, dtype=np.uint8)

# plt.figure(5)
# plt.imshow(g_view, 'gray')