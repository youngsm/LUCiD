//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file PhotonSim/src/StackingAction.cc
/// \brief Implementation of the PhotonSim::StackingAction class

#include "StackingAction.hh"

#include "G4OpticalPhoton.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

namespace PhotonSim
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track* track)
{
  // Particles created after 10 microseconds (nuclear de-excitations, long-lived decays) are
  // physically correct but irrelevant for neutrino physics and contaminate timing. Killing them
  // here, before their first step, keeps any of their light out of the event.
  if (track->GetDefinition() != G4OpticalPhoton::OpticalPhotonDefinition() &&
      track->GetGlobalTime() > 10000.0 * ns) {
    return fKill;
  }
  return fUrgent;
}

}  // namespace PhotonSim

