//This project is licensed under the terms of the Creative Commons CC BY-NC-ND 3.0 license.

//--------------------------------------------------------------------------------------------------
// Artifical Dissipation
//--------------------------------------------------------------------------------------------------

// Computes the high order dissipation for the vector type field `d_field_b` at point (ix, iy, iz)
inline REAL4 high_order_dissipation(uint ix, uint iy, uint iz, global REAL4 *d_field_b) {

	uint idx = calc_idx(ix,iy,iz);

  int bound_x = 0;
  int bound_y = 0;
  int bound_z = 0;

  for (uint i = 0; i < NUM_BOUNDS; i++) {
    bound_x = bound_x + (NUM_BOUNDS - i) * (check_bound_xr(ix, i + 1) - check_bound_l(ix, i + 1));
    bound_y = bound_y + (NUM_BOUNDS - i) * (check_bound_yr(iy, i + 1) - check_bound_l(iy, i + 1));
    bound_z = bound_z + (NUM_BOUNDS - i) * (check_bound_zr(iz, i + 1) - check_bound_l(iz, i + 1));
  }

	REAL4 HO_diss = (REAL4) {0, 0, 0, 0};

	for (int i = -STENCIL_WIDTH_HOD + 1; i < 1; i++) {
		HO_diss = HO_diss + diff_HOD_x(ix, iy, iz, bound_x, i, d_field_b)
                      + diff_HOD_y(ix, iy, iz, bound_y, i, d_field_b)
                      + diff_HOD_z(ix, iy, iz, bound_z, i, d_field_b);
	}

	HO_diss.w = 0;

	return HO_diss;
}

// Computes the first order dissipation for the vector type field `d_field_b` at point (ix, iy, iz)
// `d_field_u` is needed to compute the weights.
inline REAL4 first_order_dissipation(uint ix, uint iy, uint iz, global REAL4 *d_field_b, global REAL4 *d_field_u) {

	uint idx = calc_idx(ix,iy,iz);

	int bound_x = 0;
	int bound_y = 0;
	int bound_z = 0;

	for (uint i = 0; i < NUM_BOUNDS; i++) {
		 bound_x = bound_x + (NUM_BOUNDS - i)*(check_bound_xr(ix,i+1) - check_bound_l(ix,i+1));
		 bound_y = bound_y + (NUM_BOUNDS - i)*(check_bound_yr(iy,i+1) - check_bound_l(iy,i+1));
		 bound_z = bound_z + (NUM_BOUNDS - i)*(check_bound_zr(iz,i+1) - check_bound_l(iz,i+1));
	}

	REAL4 FO_diss = (REAL4) {0, 0, 0, 0};
  REAL4 tmp;

  // x direction
  tmp = (REAL4) {0, 0, 0, 0};
  for (uint i = 0; i < STENCIL_WIDTH; i++) {
		tmp.xyz = tmp.xyz
      + fabs(SBP_diff[NUM_BOUNDS + bound_x][i])
        * sigma(get_Field(ix,iy,iz,(i - (STENCIL_WIDTH - 1)/2), 0, 0, d_field_u), get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * (get_Field(ix,iy,iz,(i - (STENCIL_WIDTH - 1)/2), 0, 0, d_field_b).xyz - get_Field(ix,iy,iz, 0, 0, 0, d_field_b).xyz);
	}
  FO_diss = FO_diss + tmp / (REAL)(DX);

  // y direction
  tmp = (REAL4) {0, 0, 0, 0};
  for (uint i = 0; i < STENCIL_WIDTH; i++) {
		tmp.xyz = tmp.xyz
      + fabs(SBP_diff[NUM_BOUNDS + bound_y][i])
        * sigma(get_Field(ix,iy,iz, 0,(i - (STENCIL_WIDTH - 1)/2), 0, d_field_u), get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * (get_Field(ix,iy,iz, 0,(i - (STENCIL_WIDTH - 1)/2), 0, d_field_b).xyz - get_Field(ix,iy,iz, 0, 0, 0, d_field_b).xyz);
	}
  FO_diss = FO_diss + tmp / (REAL)(DY);

  // z direction
  tmp = (REAL4) {0, 0, 0, 0};
  for (uint i = 0; i < STENCIL_WIDTH; i++) {
		tmp.xyz = tmp.xyz
      + fabs(SBP_diff[NUM_BOUNDS + bound_z][i])
        * sigma(get_Field(ix,iy,iz, 0, 0,(i - (STENCIL_WIDTH - 1)/2), d_field_u), get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * (get_Field(ix,iy,iz, 0, 0,(i - (STENCIL_WIDTH - 1)/2), d_field_b).xyz - get_Field(ix,iy,iz, 0, 0, 0, d_field_b).xyz);
	}
  FO_diss = FO_diss + tmp / (REAL)(DZ);


	return FO_diss;
}

// Computes the adaptive dissipation for the vector type field `d_field_b` at point (ix, iy, iz)
// `d_field_u` is needed to compute the weights.
inline REAL4 adaptive_dissipation(uint ix, uint iy, uint iz, global REAL4 *d_field_b, global REAL4 *d_field_u) {

	REAL4 AD_diss = (REAL4) {0, 0, 0, 0};

#ifdef USE_ADAPTIVE_DISSIPATION
	uint idx = calc_idx(ix,iy,iz);

	int bound_x = 0;
	int bound_y = 0;
	int bound_z = 0;

	for (uint i = 0; i < NUM_BOUNDS; i++) {
		 bound_x = bound_x + (NUM_BOUNDS - i)*(check_bound_xr(ix,i+1) - check_bound_l(ix,i+1));
		 bound_y = bound_y + (NUM_BOUNDS - i)*(check_bound_yr(iy,i+1) - check_bound_l(iy,i+1));
		 bound_z = bound_z + (NUM_BOUNDS - i)*(check_bound_zr(iz,i+1) - check_bound_l(iz,i+1));
	}

	// First order dissipation with adjusted weights
  REAL4 tmp;
  tmp = (REAL4) {0, 0, 0, 0};
  for (uint i = 0; i < STENCIL_WIDTH; i++) {
		tmp.xyz = tmp.xyz
      + fabs(SBP_diff[NUM_BOUNDS + bound_x][i])
        * kappa4(abs(i - (STENCIL_WIDTH - 1)/2), (REAL)DX, get_Field(ix,iy,iz,(i - (STENCIL_WIDTH - 1)/2), 0, 0, d_field_b),
                                                           get_Field(ix,iy,iz, 0, 0, 0, d_field_b),
                                                           get_Field(ix,iy,iz,(i - (STENCIL_WIDTH - 1)/2), 0, 0, d_field_u),
                                                           get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * sigma(get_Field(ix,iy,iz,(i - (STENCIL_WIDTH - 1)/2), 0, 0, d_field_u), get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * (get_Field(ix,iy,iz,(i - (STENCIL_WIDTH - 1)/2), 0, 0, d_field_b).xyz - get_Field(ix,iy,iz, 0, 0, 0, d_field_b).xyz);
	}
  AD_diss.xyz = AD_diss.xyz + tmp.xyz / (REAL)(DX);

  tmp = (REAL4) {0, 0, 0, 0};
  for (uint i = 0; i < STENCIL_WIDTH; i++) {
		tmp.xyz = tmp.xyz
      + fabs(SBP_diff[NUM_BOUNDS + bound_y][i])
        * kappa4(abs(i - (STENCIL_WIDTH - 1)/2), (REAL)DY, get_Field(ix,iy,iz, 0,(i - (STENCIL_WIDTH - 1)/2), 0, d_field_b),
                                                           get_Field(ix,iy,iz, 0, 0, 0, d_field_b),
                                                           get_Field(ix,iy,iz, 0,(i - (STENCIL_WIDTH - 1)/2), 0, d_field_u),
                                                           get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * sigma(get_Field(ix,iy,iz, 0,(i - (STENCIL_WIDTH - 1)/2), 0, d_field_u), get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * (get_Field(ix,iy,iz, 0,(i - (STENCIL_WIDTH - 1)/2), 0, d_field_b).xyz - get_Field(ix,iy,iz, 0, 0, 0, d_field_b).xyz);
	}
  AD_diss.xyz = AD_diss.xyz + tmp.xyz / (REAL)(DY);

  tmp = (REAL4) {0, 0, 0, 0};
  for (uint i = 0; i < STENCIL_WIDTH; i++) {
		tmp.xyz = tmp.xyz
      + fabs(SBP_diff[NUM_BOUNDS + bound_z][i])
        * kappa4(abs(i - (STENCIL_WIDTH - 1)/2),(REAL)DZ, get_Field(ix,iy,iz, 0, 0,(i - (STENCIL_WIDTH - 1)/2), d_field_b),
                                                          get_Field(ix,iy,iz, 0, 0, 0, d_field_b),
                                                          get_Field(ix,iy,iz, 0, 0,(i - (STENCIL_WIDTH - 1)/2), d_field_u),
                                                          get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * sigma(get_Field(ix,iy,iz, 0, 0,(i - (STENCIL_WIDTH - 1)/2), d_field_u), get_Field(ix,iy,iz, 0, 0, 0, d_field_u)).xyz
        * (get_Field(ix,iy,iz, 0, 0,(i - (STENCIL_WIDTH - 1)/2), d_field_b).xyz - get_Field(ix,iy,iz, 0, 0, 0, d_field_b).xyz);
	}
  AD_diss.xyz = AD_diss.xyz + tmp.xyz / (REAL)(DZ);

	// High order dissipation with adjusted weights
	for( int term = -STENCIL_WIDTH_HOD + 1; term <= 0; term++) {
		AD_diss.xyz = AD_diss.xyz
      + (1 - kappa4(1, (REAL)DX, get_Field(ix,iy,iz,term + STENCIL_WIDTH_HOD - 1, 0, 0, d_field_b),
                                 get_Field(ix,iy,iz,term + STENCIL_WIDTH_HOD - 2, 0, 0, d_field_b),
                                 get_Field(ix,iy,iz,term + STENCIL_WIDTH_HOD - 1, 0, 0, d_field_u),
                                 get_Field(ix,iy,iz,term + STENCIL_WIDTH_HOD - 2, 0, 0, d_field_u)).xyz)
        * diff_HOD_x(ix, iy, iz, bound_x, term, d_field_b).xyz
      + (1 - kappa4(1, (REAL)DY, get_Field(ix,iy,iz, 0,term + STENCIL_WIDTH_HOD - 1, 0, d_field_b),
                                 get_Field(ix,iy,iz, 0,term + STENCIL_WIDTH_HOD - 2, 0, d_field_b),
                                 get_Field(ix,iy,iz, 0,term + STENCIL_WIDTH_HOD - 1, 0, d_field_u),
                                 get_Field(ix,iy,iz, 0,term + STENCIL_WIDTH_HOD - 2, 0, d_field_u)).xyz)
        * diff_HOD_y(ix, iy, iz, bound_y, term, d_field_b).xyz
      + (1 - kappa4(1, (REAL)DZ, get_Field(ix,iy,iz, 0, 0,term + STENCIL_WIDTH_HOD - 1, d_field_b),
                                 get_Field(ix,iy,iz, 0, 0,term + STENCIL_WIDTH_HOD - 2, d_field_b),
                                 get_Field(ix,iy,iz, 0, 0,term + STENCIL_WIDTH_HOD - 1, d_field_u),
                                 get_Field(ix,iy,iz, 0, 0,term + STENCIL_WIDTH_HOD - 2, d_field_u)).xyz)
        * diff_HOD_z(ix, iy, iz, bound_z, term, d_field_b).xyz;
	}

	AD_diss.w = 0;

#endif

	return AD_diss;
}


// Switch between available artificial dissipation options
inline REAL4 artificial_dissipation(uint ix, uint iy, uint iz, global REAL4 *d_field_b, global REAL4 *d_field_u) {

	REAL4 diss = (REAL4) {0, 0, 0, 0};

	#ifdef USE_HIGH_ORDER_DISSIPATION
    diss = high_order_dissipation(ix, iy, iz, d_field_b);
  #elif defined USE_FIRST_ORDER_DISSIPATION
    diss = first_order_dissipation(ix, iy, iz, d_field_b, d_field_u);
  #elif defined USE_ADAPTIVE_DISSIPATION
    diss = adaptive_dissipation(ix, iy, iz, d_field_b, d_field_u);
  #else
    #warning "Warning in artificial_dissipation.cl: No artifical dissipation is added!"
  #endif

	return diss;
}
