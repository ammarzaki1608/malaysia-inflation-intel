-- NOTE: despite the model/column name, this is NOT the Overnight Policy Rate.
-- BNM's interest_rates dataset has no OPR series; rate_type 'br' is the
-- commercial banks' average Base Rate (published from ~2022), which follows the
-- OPR. Name kept to avoid breaking downstream models; treat as "base rate".
with source as (

    select * from {{ ref('stg_interest_rates')}}

),

opr_only as (

    select
        date,
        rate_value as opr_rate

    from source
    where rate_type = 'br'
        and bank = 'commercial'

)

select * from opr_only