//This project is licensed under the terms of the Creative Commons CC BY-NC-ND 3.0 license.

// Contains deriviative operator and projector for divergence cleaning

//--------------------------------------------------------------------------------------------------
// Derivative Operator
//--------------------------------------------------------------------------------------------------

// Computes the second derivativ af the scalar field `d_field` at point (ix,iy,iz).
// The function is customized for divergence cleaning.
inline REAL laplace_divcleaning(uint ix, uint iy, uint iz, global REAL *d_field) {

	REAL val = 0;

  int bound_x = 0;
  int bound_y = 0;
  int bound_z = 0;

  #ifdef USE_LAPLACE_WIDE_STENCIL_DIRICHLET

	  for (uint i = 0; i < NUM_BOUNDS_LAPLACE_WS_D0; i++) {
      bound_x = bound_x + (NUM_BOUNDS_LAPLACE_WS_D0 - i)*(check_bound_xr(ix,i+1) - check_bound_l(ix,i+1));
      bound_y = bound_y + (NUM_BOUNDS_LAPLACE_WS_D0 - i)*(check_bound_yr(iy,i+1) - check_bound_l(iy,i+1));
		  bound_z = bound_z + (NUM_BOUNDS_LAPLACE_WS_D0 - i)*(check_bound_zr(iz,i+1) - check_bound_l(iz,i+1));
	  }
    for (uint i = 0; i < STENCIL_WIDTH_LAPLACE_WS_D0; i++) {
		  val = val + SBP_laplace_WS_D0[NUM_BOUNDS_LAPLACE_WS_D0 + bound_x][i]*get_Field_S(ix,iy,iz,(i - (STENCIL_WIDTH_LAPLACE_WS_D0 - 1)/2),0,0,d_field)/((REAL)DX*(REAL)DX)
                + SBP_laplace_WS_D0[NUM_BOUNDS_LAPLACE_WS_D0 + bound_y][i]*get_Field_S(ix,iy,iz,0,(i - (STENCIL_WIDTH_LAPLACE_WS_D0 - 1)/2),0,d_field)/((REAL)DY*(REAL)DY)
                + SBP_laplace_WS_D0[NUM_BOUNDS_LAPLACE_WS_D0 + bound_z][i]*get_Field_S(ix,iy,iz,0,0,(i - (STENCIL_WIDTH_LAPLACE_WS_D0 - 1)/2),d_field)/((REAL)DZ*(REAL)DZ);
	  }

  #elif defined USE_LAPLACE_WIDE_STENCIL_LNS

	  for (uint i = 0; i < NUM_BOUNDS_LAPLACE_WS_LN; i++) {
      bound_x = bound_x + (NUM_BOUNDS_LAPLACE_WS_LN - i)*(check_bound_xr(ix,i+1) - check_bound_l(ix,i+1));
      bound_y = bound_y + (NUM_BOUNDS_LAPLACE_WS_LN - i)*(check_bound_yr(iy,i+1) - check_bound_l(iy,i+1));
		  bound_z = bound_z + (NUM_BOUNDS_LAPLACE_WS_LN - i)*(check_bound_zr(iz,i+1) - check_bound_l(iz,i+1));
	  }
    for (uint i = 0; i < STENCIL_WIDTH_LAPLACE_WS_LN; i++) {
		  val = val + SBP_laplace_WS_LN[NUM_BOUNDS_LAPLACE_WS_LN + bound_x][i]*get_Field_S(ix,iy,iz,(i - (STENCIL_WIDTH_LAPLACE_WS_LN - 1)/2),0,0,d_field)/((REAL)DX*(REAL)DX)
                + SBP_laplace_WS_LN[NUM_BOUNDS_LAPLACE_WS_LN + bound_y][i]*get_Field_S(ix,iy,iz,0,(i - (STENCIL_WIDTH_LAPLACE_WS_LN - 1)/2),0,d_field)/((REAL)DY*(REAL)DY)
                + SBP_laplace_WS_LN[NUM_BOUNDS_LAPLACE_WS_LN + bound_z][i]*get_Field_S(ix,iy,iz,0,0,(i - (STENCIL_WIDTH_LAPLACE_WS_LN - 1)/2),d_field)/((REAL)DZ*(REAL)DZ);
	  }

  #elif defined USE_LAPLACE_NARROW_STENCIL_DIRICHLET

	  for (uint i = 0; i < NUM_BOUNDS_LAPLACE_NS_D0; i++) {
      bound_x = bound_x + (NUM_BOUNDS_LAPLACE_NS_D0 - i)*(check_bound_xr(ix,i+1) - check_bound_l(ix,i+1));
      bound_y = bound_y + (NUM_BOUNDS_LAPLACE_NS_D0 - i)*(check_bound_yr(iy,i+1) - check_bound_l(iy,i+1));
		  bound_z = bound_z + (NUM_BOUNDS_LAPLACE_NS_D0 - i)*(check_bound_zr(iz,i+1) - check_bound_l(iz,i+1));
	  }
    for (uint i = 0; i < STENCIL_WIDTH_LAPLACE_NS_D0; i++) {
		  val = val + SBP_laplace_NS_D0[NUM_BOUNDS_LAPLACE_NS_D0 + bound_x][i]*get_Field_S(ix,iy,iz,(i - (STENCIL_WIDTH_LAPLACE_NS_D0 - 1)/2),0,0,d_field)/((REAL)DX*(REAL)DX)
                + SBP_laplace_NS_D0[NUM_BOUNDS_LAPLACE_NS_D0 + bound_y][i]*get_Field_S(ix,iy,iz,0,(i - (STENCIL_WIDTH_LAPLACE_NS_D0 - 1)/2),0,d_field)/((REAL)DY*(REAL)DY)
                + SBP_laplace_NS_D0[NUM_BOUNDS_LAPLACE_NS_D0 + bound_z][i]*get_Field_S(ix,iy,iz,0,0,(i - (STENCIL_WIDTH_LAPLACE_NS_D0 - 1)/2),d_field)/((REAL)DZ*(REAL)DZ);
	  }

	#else
	 	#error "Define discretisation if the Laplace operator for divergence cleaning!\n"
  #endif

  return val;
}

// Computes the divergence af the vector field `d_field` and stores the result in `d_field_div`.
// The function is customized for divergence cleaning.
kernel void calc_div_divcleaning(global REAL4 *d_field, global REAL *d_field_div) {

  uint ix = get_global_id(0) + BNODES;
  uint iy = get_global_id(1) + BNODES;
  uint iz = get_global_id(2) + BNODES;

  uint idx = calc_idx(ix,iy,iz);

  // Calc divergence
  d_field_div[idx] = div(ix, iy, iz, d_field);
}


// Computes the second derivatice af the scalar field `d_field` and stores the result in `d_field_laplace`.
// The function is customized for divergence cleaning.
kernel void calc_laplace_divcleaning(global REAL *d_field, global REAL *d_field_laplace) {

  uint ix = get_global_id(0) + BNODES;
  uint iy = get_global_id(1) + BNODES;
  uint iz = get_global_id(2) + BNODES;

  uint idx = calc_idx(ix,iy,iz);

  d_field_laplace[idx] = laplace_divcleaning(ix, iy, iz, d_field);
}


//--------------------------------------------------------------------------------------------------
// Projector
//--------------------------------------------------------------------------------------------------


// Uses Φ to project the magnetic field onto the space of divergence free vector fields.
// The function is customized for divergence cleaning.

// TODO: .xyz or all components?
kernel void projector_divcleaning(global REAL4 *d_field_b, global REAL *d_field_phi) {

  uint ix = get_global_id(0);
  uint iy = get_global_id(1);
  uint iz = get_global_id(2);

  uint idx = calc_idx(ix,iy,iz);

  #ifdef USE_LAPLACE_WIDE_STENCIL_LNS
    d_field_b[idx].xyz = d_field_b[idx].xyz - grad_adj(ix,iy, iz, d_field_phi).xyz;
  #else
    d_field_b[idx].xyz = d_field_b[idx].xyz + grad(ix,iy, iz, d_field_phi).xyz;
  #endif
}
