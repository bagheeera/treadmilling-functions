log                 log.txt
units               lj
dimension           2
atom_style          molecular
read_data           configuration.txt extra/bond/per/atom 5  extra/special/per/atom 20  extra/angle/per/atom 3

variable            tscale equal 2 ## time is in step*${tstep}/tscale seconds
variable            ron equal 8.0/${tscale}                                # growth rate [monomers/s]
variable            rdis equal 10.0/${tscale}                               # dissociation rate [monomers/s]
variable            rnuc equal 10.00000/${tscale}                    # nucleation rate [filaments/s]
variable            tauhyd equal 1.0*${tscale}                          # hydrolysis time [seconds]
variable            Kbond equal 1000.0                               # bond constant [kT/sigma2]
variable            tstep equal 0.001                                 # simulation timestep size [seconds]
variable            realtime equal step*${tstep}/${tscale}            # simulation time in real units [seconds]
variable            run_time equal 2000*${tscale}                            # simulation run time [seconds]
variable            treact equal 0.010000*${tscale}                            # reaction attempt time [seconds]
variable            tnuc equal *${tscale}                           # nucleation reaction attempt time [seconds]

variable            modtime equal 0.0                         # modulation time [seconds]
variable            condmod equal "v_realtime > v_modtime"
variable            ratesratio equal 1.0

variable            frame_rate equal 10*${tscale}                          # coarse dumping interval [seconds]
variable            seed equal 1234                                  # random number generator seed
variable            maxatoms equal 0+4872                              # maximum number of atoms allowed in the system
variable            fCurv equal 0.0                                 # magnitude of the curvature force [kT/sigma]

variable            condarr equal 1 # don't arrest

variable            poff equal "1.0*v_condarr"                    # shrinking reaction initiation probability (0 or 1)
variable            condatoms equal "atoms >= v_maxatoms+2"
variable            pon equal 1 #(1-v_condatoms)                     # growing reaction initiation probability (0 or 1)

## synthase reactions
variable            pact equal 0 ## 5 to 6

## switch on synth-filament interactions after given time
variable            condattract equal "v_realtime >= 600"
variable            pdeact equal "1.0*v_condattract"  ## 6 to 5

variable            psynthbond equal 1.00000

## box modulation
variable  kswitch equal 0.2
variable rswitch1 equal 40
variable rswitch2 equal -40
variable s1 atom 1/(1 + exp(v_kswitch*(y - v_rswitch1)))
variable s2 atom 1/(1 + exp(v_kswitch*(y - v_rswitch2)))
variable bumpy atom v_s1 * (1 - v_s2)
variable sx1 atom 1/(1 + exp(v_kswitch*(x - v_rswitch1)))
variable sx2 atom 1/(1 + exp(v_kswitch*(x - v_rswitch2)))
variable bumpx atom v_sx1 * (1 - v_sx2)
variable kon atom v_ron*${treact}*v_ratesratio*(1-v_condmod) + v_ron*${treact}*v_bumpy*v_bumpx*v_condmod

variable            knuc atom v_ratesratio                        # nucleation probability
variable            pnuc0 equal v_pon*(1-v_condmod)
variable            pnuc1 equal v_pon*v_condmod
variable            kdis equal ${rdis}*${treact}                          # dissociation probability

variable            kdis equal ${rdis}*${treact}                          # dissociation probability

variable            thyd equal ${tauhyd}/${tstep}                 # hydrolysis time [simulation steps]
variable            rstep equal ${treact}/${tstep}                      # reaction interval [simulation steps]
variable            nucstep equal ${tnuc}/${tstep}                      # nucleation reaction interval [simulation steps]
variable            run_steps equal ${run_time}/${tstep}          # simulation run time [simulation steps]
variable            dump_time equal ${frame_rate}/${tstep}        # dumping interval [simulation steps]


variable            stab_steps equal 1
variable            synth_stab_steps equal 10 ## for synth binding on reaction

group               ghosts type 4

special_bonds       lj 1.0 1.0 1.0
bond_style          hybrid harmonic #morse 
bond_coeff          1 harmonic ${Kbond} 1.0


variable synthrmin equal 0.50000
variable synthrmax equal 2.00000

angle_style         harmonic
angle_coeff         1 10000 180.0
angle_coeff         2 100.0 180.0

pair_style          hybrid/overlay zero 1.50 cosine/squared 1.50 soft 1.0
pair_coeff          * * cosine/squared 0.00 1.00 1.10
pair_coeff          * * zero 2
pair_coeff          5 7 zero 2
pair_coeff          1 5 zero 2
pair_coeff          3 5 zero 2
pair_coeff          1 6 zero 2
pair_coeff          1 5 cosine/squared 10 0.5 0.8 wca
pair_coeff          2 5 cosine/squared 10 0.5 0.8 wca
pair_coeff          3 5 cosine/squared 10 0.5 0.8 wca
pair_coeff          5 5 soft 5 1
pair_coeff          6 6 soft 5 1
pair_coeff          5 6 soft 5 1

pair_coeff          1 1 cosine/squared 10 1.00  wca
pair_coeff          1 2 cosine/squared 1.00 1.00 1.00 wca
pair_coeff          1 3 cosine/squared 1.00 1.00 1.00 wca
pair_coeff          2 2 cosine/squared 1.00 1.00 1.00 wca
pair_coeff          2 3 cosine/squared 1.00 1.00 1.00 wca
pair_coeff          3 3 cosine/squared 1.00 1.00 1.00 wca

#neigh_modify        exclude molecule/intra all

# Nucleation Reactions_rdis molecular templates
molecule            mPreNucleation /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_Nucleation.txt
molecule            mPostNucleation /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_Nucleation.txt

# Growth Reactions_rdis molecular templates
molecule            mPreDimerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_DimerOn.txt
molecule            mPostDimerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_DimerOn.txt
molecule            mPreTrimerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_TrimerOn.txt
molecule            mPostTrimerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_TrimerOn.txt
molecule            mPreOligomerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_OligomerOn.txt
molecule            mPostOligomerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_OligomerOn.txt

# Shrink Reactions_rdis molecular templates
molecule            mPreDimerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_DimerOff.txt
molecule            mPostDimerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_DimerOff.txt
molecule            mPreTrimerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_TrimerOff.txt
molecule            mPostTrimerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_TrimerOff.txt
molecule            mPreQuartomerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_QuartomerOff.txt
molecule            mPostQuartomerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_QuartomerOff.txt
molecule            mPreOligomerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_OligomerOff.txt
molecule            mPostOligomerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_OligomerOff.txt

# div templates
molecule mPreDiviAct /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_div_act.txt
molecule mPostDiviAct /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_div_act.txt
molecule mPreDEact /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/pre_DEact.txt
molecule mPostDEact /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/post_DEact.txt

variable            vCreationSteps atom i_creation_steps
variable            vHydrolysisRn atom d_hydrolysis_rn
fix freact all bond/react  stabilization yes AllAtoms 0.1  reset_mol_ids molmap lifetime hydrolysis ${seed} &
                react Nucleation all ${nucstep} 0.900000 1.100000 mPreNucleation mPostNucleation /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_Nucleation.txt prob v_pon ${seed} stabilize_steps ${stab_steps} modify_create overlap 1 modify_create nuc yes         &
                react DimerOn all ${rstep} 0.900000 1.100000 mPreDimerOn mPostDimerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_DimerOn.txt prob v_pon ${seed} stabilize_steps ${stab_steps} modify_create fit 1 modify_create overlap 1         &
                react TrimerOn all ${rstep} 0.900000 1.100000 mPreTrimerOn mPostTrimerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_TrimerOn.txt prob v_pon ${seed} stabilize_steps ${stab_steps} modify_create fit 1 modify_create overlap 1         &
                react OligomerOn all ${rstep} 0.900000 1.100000 mPreOligomerOn mPostOligomerOn /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_OligomerOn.txt prob v_pon ${seed} stabilize_steps ${stab_steps} modify_create fit 1 modify_create overlap 1         &
                react OligomerOff all ${rstep} 0.900000 1.100000 mPreOligomerOff mPostOligomerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_OligomerOff.txt prob v_poff ${seed} stabilize_steps ${stab_steps}         &
                react QuartomerOff all ${rstep} 0.900000 1.100000 mPreQuartomerOff mPostQuartomerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_QuartomerOff.txt prob v_poff ${seed} stabilize_steps ${stab_steps}         &
                react TrimerOff all ${rstep} 0.900000 1.100000 mPreTrimerOff mPostTrimerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_TrimerOff.txt prob v_poff ${seed} stabilize_steps ${stab_steps}         &
                react DimerOff all ${rstep} 0.900000 1.100000 mPreDimerOff mPostDimerOff /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_DimerOff.txt prob v_poff ${seed} stabilize_steps ${stab_steps} &
				react diviact  &
					all  ${rstep}  &
					0.100000 10  &
					mPreDiviAct mPostDiviAct  /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_DivAct.txt &
					prob v_pact ${seed} &
				react divideact &
					all ${rstep}   &
					0.000000 9999 &
					mPreDEact mPostDEact /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/molid/map_DEact.txt &
					prob v_pdeact ${seed} 

variable            vMaskHeadType atom "type==3"
group               HeadMons dynamic all var vMaskHeadType every 1
variable            vMaskTailType atom "type==2"
group               TailMons dynamic all var vMaskTailType every 1
variable            vMaskHTType atom "type==2 || type==3"
group               HTMons dynamic all var vMaskHTType every 1
variable            vMaskAlive atom "type==1 || type==2 || type==3"
group               alive dynamic all var vMaskAlive every 1
variable vStep atom "step"
fix      fStep all store/state 1 v_vStep


fix                 fLang all langevin 1.0 1.0 1.0 ${seed}

## INTEGRATE ONLY NONGHOST PARTICLES
fix fNVE AllAtoms_REACT nve
variable vMaskGrid atom "type==7"
group grid dynamic all var vMaskGrid every 100
variable vMaskReactNongrid atom "(1-gmask(grid)) * gmask(AllAtoms_REACT)"
group integr dynamic all var vMaskReactNongrid every 1
fix fNVE integr nve

#fix fNVE AllAtoms_REACT nve

## freeze grid particles
group grid type 7
fix freeze grid setforce 0 0 0
velocity grid set 0 0 0



## DONT OUTPUT GRID PARTICLES
variable vnongrid atom "(1-gmask(grid))"
group nongrid dynamic all var vnongrid every ${dump_time}
dump 1 nongrid custom ${dump_time} output.xyz v_vStep id mol type x y 
dump_modify 1 format line "%6.f %d %d %d %.1f %.1f" #  %.1f %.1f 




thermo              ${dump_time}
compute_modify      thermo_temp dynamic/dof yes
thermo_style        custom step v_realtime temp pe ke etotal epair ebond eangle press vol density atoms


## bonds output
#compute             cBonds all property/local batom1 batom2 btype
#compute             cBondDxys all bond/local engpot force dist
#dump 2 all local ${dump_time} bonds.dump c_cBonds[*] c_cBondDxys[*]
#dump_modify 2 format line "%.0f %.0f %.0f %.3f %.3f %.3f"

fix                 twodim all enforce2d

variable            restartsteps equal v_run_steps/20
restart             ${restartsteps} restart

timestep            ${tstep}
run                 ${run_steps}