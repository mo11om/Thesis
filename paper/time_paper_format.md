| | |
| :--- | :--- |
| **Context** | our provision for gross - to - net allowances was $ 3. 0 million at march 31, 2024, $ [0. 6] (Value: 0.6) million of which was recorded as a reduction to accounts receivable and $ 2. 4 million recorded as a component of accrued expenses. 10 - q ; 2024 - 03 - 31 ; 2024 ; q1 none none |
| **g_task** | 0.5866290331 |
| **Label** | instant; future |
| **Pred.** | period; future |
| **Expert Label** | instant; current |
| **Reason** | The $0.6 million value represents a reduction to accounts receivable at a specific point in time, classifying its structural period type as an instant metric. Because this exact snapshot date (March 31, 2024) aligns perfectly with the document's reporting period end date, its relative temporal context is mapped as current. |

---

| | |
| :--- | :--- |
| **Context** | in december 2019, as a result of the initial stipulation and settlement agreement, aep texas ( a ) recorded an impairment of $ [33] (Value: 33.0) million related to capital investments, which included $ 10 million of 2019 investments, in asset impairments and other related charges on the statements of income, ( b ) recorded a $ 30 million provision for refund on the statements of income for revenues previously collected through rates and ( c ) wrote - off $ 4 million of rate case expenses to other operation on the statements of income. 10 - q ; 2020 - 06 - 30 ; 2020 ; q2 none none |
| **g_task** | 0.5713546872 |
| **Label** | period; current_future |
| **Pred.** | period; future |
| **Expert Label** | duration:past |
| **Reason** |  The $33 million is classified as a duration because it is an impairment charge recognized on the income statement, representing an activity over a period rather than a frozen balance sheet snapshot. It receives the past label because the charge was recorded in December 2019, which historically predates the document's primary Q2 2020 reporting period.|

---

| | |
| :--- | :--- |
| **Context** | includes current portion of other liabilities of $ [5] (Value: 5.0) million at march 31, 2022 and $ 4 million at march 31, 2021. 10 - q ; 2022 - 03 - 31 ; 2022 ; q1 none none |
| **g_task** | 0.5713546872 |
| **Label** | period; current_future |
| **Pred.** | period; future |
| **Expert Label** | instant: past |
| **Reason** | The $5 million value is a balance sheet liability measured at the specific, single date of March 31, 2022, which defines its period type as an **instant**. Because this exact measurement date matches the end date of the filing's current 10-Q accounting period, its temporal classification is **current**. |

 
---

| | |
| :--- | :--- |
| **Context** | at june 30, 2023, the company had an outstanding balance of $ [25. 0] (Value: 25.0) million against its 2023 revolving facility. 10 - k ; 2023 - 06 - 30 ; 2023 ; fy during the fiscal year ended june 30, 2021, the company repaid $ 55. 0 million against its 2019 revolving facility that was outstanding as of june 30, 2020 and had no outstanding balances as of june 30, 2021 and 2022. the company has $ 110. 2 million availability under the 2023 revolving facility as of june 30, 2023. |
| **g_task** | 0.5713546872 |
| **Label** | period; future |
| **Pred.** | period; future |
| **Expert Label** | instant:current |
| **Reason** |  The $25.0 million outstanding facility balance represents a balance sheet snapshot taken at a single, specific point in time on June 30, 2023. As this specific date aligns exactly with the end date of the current 10-K fiscal year filing, it directly corresponds to the current accounting period.|

---

| | |
| :--- | :--- |
| **Context** | the company had proceeds from the sale of trade receivables under the amended arrangement of $ 70. 0 and $ 197. 7, respectively, for the three and nine months ended june 30, 2024 ( $ [0. 0] (Value: 0.0) for the three and nine months ended june 30, 2023 ). 10 - q ; 2024 - 06 - 30 ; 2024 ; q3 in accordance with asc 869, transfers and servicing, this amended arrangement is deemed a true sale, as the company retains no rights or interest and has no obligations with respect to the trade receivables. as of june 30, 2024 and september 30, 2023, trade receivables in the amount of $ 32. 1and $ 0. 00. 0, respectively, were sold to the financial institution and are not reflected in the trade receivables in the consolidated balance sheets. |
| **g_task** | 0.5553240776 |
| **Label** | period; future |
| **Pred.** | period; past_current |
| **Expert Label** | duration:past |
| **Reason** | The target value of $0.0 measures proceeds explicitly over a continuous time span ("the three and nine months ended June 30, 2023"), clearly defining the period type as a duration. Since this specific historical duration concluded a full year prior to the current 10-Q accounting period ending June 30, 2024, it is classified temporally as past. |

---

| | |
| :--- | :--- |
| **Context** | the additional payments will be based on a percent of net invoices for which payments have been received on systems sold to electric vehicle ( " ev " ) or battery customers in excess of cad $ [2, 500] (Value: 2500.0) per year in each of the five years. 10 - q ; 2022 - 06 - 30 ; 2022 ; q2 in addition, we may pay the seller up to an additional cad $ 5, 000 in the five - year period from 2022 through 2026. the maximum payment is capped at cad $ 5, 000, which equates to approximately $ 3, 900 at june 30, 2022. |
| **g_task** | 0.5536586046 |
| **Label** | period; past_future |
| **Pred.** | period; future |
| **Expert Label** | duration:current_future |
| **Reason** | The primary narrative outlines a contingent payment structure over a "five-year period from 2022 through 2026," which classifies as a period: current_future because it spans from the current 10-Q reporting period (Q2 2022) into subsequent fiscal years. Additionally, the text contains an instant: current measurement for the maximum payment equivalent of $3,900, as it is valued at the specific, single point in time ending the current accounting period (June 30, 2022). |

---

| | |
| :--- | :--- |
| **Context** | as of march 31, 2021 and december 31, 2020 there are estimated insurance recovery receivables of $ 1, 159 and $ [1, 025] (Value: 1025.0) in " self - insurance receivable ", respectively. 10 - q ; 2021 - 03 - 31 ; 2021 ; q1 accordingly, the estimated insurance recovery receivables are included within " self - insurance receivables " on the consolidated balance sheet. none |
| **g_task** | 0.5536586046 |
| **Label** | period; past_future |
| **Pred.** | period; future |
| **Expert Label** | instant:past |
| **Reason** | The targeted value of $1,025 represents an insurance recovery receivable measured at a specific, single point in time ("as of December 31, 2020"), establishing its period type as an instant. Because the associated filing is a Q1 2021 10-Q (ending March 31, 2021), the December 31, 2020 date occurs entirely before the current three-month accounting period, designating its temporal classification as past. |

---

| | |
| :--- | :--- |
| **Context** | the impact of the measurement period adjustments to our results of operations resulted in increases to previously reported depreciation and amortization expense of $ [2. 0] (Value: 2.0) million in 2021. 10 - q ; 2021 - 06 - 30 ; 2021 ; q2 these adjustments in fair value also resulted in an increase to the deferred tax liability of $ 9. 7 million. none |
| **g_task** | 0.5307079554 |
| **Label** | instant; current |
| **Pred.** | instant; future |
| **Expert Label** | duration:past |
| **Reason** |  Because the text discusses depreciation and amortization expenses, it represents a continuous duration of measurement rather than a single point in time, classifying the period type as a "period." Furthermore, since the document is a Q2 2021 10-Q (where the "current" period is the specific three-month window ending June 30), the reference to "previously reported" 2021 expenses indicates an adjustment to the Q1 timeframe, which occurred entirely before the current accounting period.|

---

| | |
| :--- | :--- |
| **Context** | goodwill was $ 680. 6 million and $ [671. 9] (Value: 671.9) million, respectively, at december 31, 2020 and 2019. 10 - k ; 2020 - 12 - 31 ; 2020 ; fy none estimating the fair value of a reporting unit for goodwill impairment is highly sensitive to changes in projections and assumptions and changes in assumptions could potentially lead to impairment. |
| **g_task** | 0.5307079554 |
| **Label** | instant; current |
| **Pred.** | period; past_future |
| **Expert Label** | instant:past |
| **Reason** | The $671.9 million goodwill value is measured at a specific single point in time ("at december 31, 2019"), dictating its period type as an instant rather than a continuous duration. Because the provided context establishes the current reporting period as the 10-K fiscal year ending December 31, 2020, this prior-year balance sheet date falls entirely before the current accounting period, classifying it as past. |

---

