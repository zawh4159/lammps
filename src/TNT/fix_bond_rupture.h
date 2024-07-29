/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(bond/rupture,FixBondRupture);
// clang-format on
#else

#ifndef LMP_FIX_BOND_RUPTURE_H
#define LMP_FIX_BOND_RUPTURE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondRupture : public Fix {
 public:
  FixBondRupture(class LAMMPS *, int, char **);
  ~FixBondRupture() override;
//   void post_constructor() override;
  int setmask() override;
  void init() override;
  void post_integrate() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 protected:
  int me, nprocs;

  // Default arguments
  int btype;
  double rcritsq;

  // Flags for keywords
  int flag_mol;

  // Internal methods/functions
  void process_broken(int, int);
  void update_special();
  void update_topology();

  // Create an array to store bonds broken this timestep (new)
  // and since the last neighbor list build
  std::vector<std::pair<tagint, tagint>> new_broken_pairs;

};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix bond/create/dynamic command

Self-explanatory.

E: Invalid bond type in fix bond/create/dynamic command

Self-explanatory.

E: Cannot use fix bond/create/dynamic with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix bond/create/dynamic command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix bond/create/dynamic cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix bond/create/dynamic angle type is invalid

Self-explanatory.

E: Fix bond/create/dynamic dihedral type is invalid

Self-explanatory.

E: Fix bond/create/dynamic improper type is invalid

Self-explanatory.

E: Cannot yet use fix bond/create/dynamic with this improper style

This is a current restriction in LAMMPS.

E: Fix bond/create/dynamic needs ghost atoms from further away

This is because the fix needs to walk bonds to a certain distance to
acquire needed info, The comm_modify cutoff command can be used to
extend the communication range.

E: New bond exceeded bonds per atom in fix bond/create/dynamic

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix bond/create/dynamic

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

E: Fix bond/create/dynamic induced too many angles/dihedrals/impropers per atom

See the read_data command for info on setting the "extra angle per
atom", etc header values to allow for additional angles, etc to be
formed.

E: Special list size exceeded in fix bond/create/dynamic

See the read_data command for info on setting the "extra special per
atom" header value to allow for additional special values to be
stored.

W: Fix bond/create/dynamic is used multiple times or with fix bond/break - may not work as expected

When using fix bond/create/dynamic multiple times or in combination with
fix bond/break, the individual fix instances do not share information
about changes they made at the same time step and thus it may result
in unexpected behavior.

*/
