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