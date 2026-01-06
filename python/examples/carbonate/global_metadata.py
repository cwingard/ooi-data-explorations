#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Global metadata attributes shared by all datasets created in this processing
"""
SHARED = {
    'global': {
        'project': 'U.S. National Science Foundation (NSF) Ocean Observatories Initiative (OOI)',
        'institution': 'OOI Coastal Endurance Array',
        'acknowledgement': 'Funding provided by the U.S. National Science Foundation (NSF)',
        'infoUrl': 'https://oceanobservatories.org',
        'summary': ('The OOI Coastal Endurance Array is a multi-scaled array utilizing fixed and mobile assets to observe cross-shelf and '
                    'along-shelf variability in the coastal upwelling region off the Oregon and Washington coasts. The array '
                    'also provides an extensive spatial footprint that encompasses a prototypical eastern boundary current regime and '
                    'connectivity with the OOI Regional Cabled Array. OOI Coastal Glider deployments bridge the distances between the fixed sites '
                    'of the Coastal Endurance Array and allow for adaptive sampling of the coastal waters off Washington and Oregon.'),
        'creator_name': 'Christopher Wingard',
        'creator_institution': 'OOI Coastal Endurance Array',
        'creator_email': 'chris.wingard@oregonstate.edu',
        'creator_url': 'https://ceoas.oregonstate.edu/ooi/',
        'contributor_name': 'Edward Dever, Jonathan Fram, Christopher Wingard',
        'contributor_role': 'Endurance Array Principal Investigator/Project Scientist, Endurance Array Project Manager, Endurance Array Data Management',
        'publisher_name': 'NSF Ocean Observatories Initiative',
        'publisher_url': 'https://oceanobservatories.org',
        'publisher_email': 'help@oceanobservatories.org',
        'featureType': 'timeSeries',
        'cdm_data_type': 'Station',
        'Conventions': 'CF-1.7',
        'processing_level': 'Reprocessed from source data produced by OOI',
        'license': ('All OOI data, including data from OOI core sensors and all proposed sensors added by Principal Investigators, will be '
                    'rapidly disseminated, open, and freely available (within constraints of national security). Rapidly disseminated implies '
                    'that data will be made available as soon as technically feasible, but generally in near real-time, with latencies as small '
                    'as seconds for the cabled components. In limited cases, individual PIs who have developed a data source that becomes part of '
                    'the OOI network may request exclusive rights to the data for a period of no more than one year from the onset of the data '
                    'stream. The reliability, quality and completeness of data obtained through OOI are intended to be used in an education or '
                    'research context. It is assumed that outages and errors can occur and are dealt with by the users of the data. These data and '
                    'software are not for use in operational or decision-making settings. The OOI program makes reasonable efforts to ensure that '
                    'the data provided are accurate. However, there may be no Quality Control (QC) performed on data acquired and provided through '
                    'the OOI program, and there may be no Quality Assurance (QA) provided on information on those data sets. If QC/QA is performed, '
                    'it is described in the metadata. The OOI program both produces and, through collaborations within the geosciences community, '
                    'gains access to data sets which may be redistributed either directly or indirectly at no cost and with no restrictions. With '
                    'regard to data distribution, all users must comply with any applicable U.S. export laws and regulations. The OOI Program is not '
                    'responsible for the use of the data it provides. The full data policy is available at '
                    'https://oceanobservatories.org/usage-policy'),
    }
}
