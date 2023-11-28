////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

(() => {
  /**
   * GaussianSplatting Api
   */
  class GaussianSplattingApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'GaussianSplatting';

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider("gaussiansplatting.setScale", 0.01, 1, 0.01, [1.0]);
    }
  }

  CosmoScout.init(GaussianSplattingApi);
})();