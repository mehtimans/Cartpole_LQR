
#include <CANAdapter.h>
#include <functional>
#include <stdio.h>


CANAdapter::CANAdapter(std::function<void(can_frame_t*)> reception_handler)
    :reception_handler{std::move(reception_handler)},
     adapter_type(ADAPTER_NONE),
     parser(NULL)
{
    printf("CAN adapter created.\n");
}


CANAdapter::~CANAdapter()
{
    printf("Destroying CAN adapter...\n");
}


void CANAdapter::transmit(can_frame_t*)
{
    if (adapter_type == ADAPTER_NONE)
    {
        printf("Unable to send: Unspecified CAN adapter\n");
        return;
    }
}
